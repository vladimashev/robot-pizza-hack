# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from smolvla.config import SmolVLAConfig
from smolvla.flow_matching import VLAFlowMatching
from smolvla.utils import pad_vector, resize_with_pad

OBS_STATE = "observation.state"
OBS_LANG_TOKENS = "observation.language.tokens"
OBS_LANG_MASK = "observation.language.attention_mask"
ACTION = "action"
ACTION_IS_PAD = "action_is_pad"


class SmolVLAPolicy(nn.Module):
    """Minimal SmolVLA policy: flow-matching training loss + iterative inference.

    No RTC, no Aloha adaptation, no PEFT — just the core training/inference surface.
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        self.model = VLAFlowMatching(config)

    # --- training ---

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Compute the flow-matching training loss.

        Returns:
            (scalar_loss, info_dict)
        """
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANG_TOKENS]
        lang_masks = batch[OBS_LANG_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get(ACTION_IS_PAD)

        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)

        # Slice to the real (unpadded) action dim so padded dims don't dilute the loss.
        losses = losses[:, :, : self.config.action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        loss = losses.mean()
        return loss, {"loss": loss.item()}

    # --- inference ---

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Return a chunk of predicted actions, shape (B, chunk_size, action_dim)."""
        self.eval()
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANG_TOKENS]
        lang_masks = batch[OBS_LANG_MASK]

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        return actions[:, :, : self.config.action_dim]

    # --- batch preparation ---

    def prepare_images(self, batch):
        """Resize-and-pad images to the VLM input size, normalize to [-1, 1] for SigLIP."""
        images = []
        img_masks = []
        present = [k for k in self.config.image_keys if k in batch]
        missing = [k for k in self.config.image_keys if k not in batch]

        if not present:
            raise ValueError(
                f"No image features found. Expected one of {self.config.image_keys}; got {list(batch.keys())}."
            )

        img = None
        mask = None
        for key in present:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            pad_key = f"{key}_padding_mask"
            if pad_key in batch:
                mask = batch[pad_key].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for i in range(len(missing)):
            if i >= self.config.empty_cameras:
                break
            images.append(torch.ones_like(img) * -1)
            img_masks.append(torch.zeros_like(mask))
        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # --- task tokenization ---

    def tokenize_task(
        self,
        task: str | list[str],
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize task string(s) with the VLM tokenizer (SmolVLM2).

        SmolVLA was trained with a trailing '\\n' on the task string — we preserve that.
        """
        if isinstance(task, str):
            task = [task]
        task = [t if t.endswith("\n") else t + "\n" for t in task]
        tokenizer = self.model.vlm_with_expert.processor.tokenizer
        enc = tokenizer(
            task,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = enc["input_ids"]
        mask = enc["attention_mask"].bool()
        if device is not None:
            tokens = tokens.to(device)
            mask = mask.to(device)
        return tokens, mask

    # --- save / load ---

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        with open(save_directory / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)
        save_file(self.state_dict(), save_directory / "model.safetensors")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_overrides: dict | None = None,
        strict: bool = True,
        map_location: str | torch.device | None = None,
    ) -> "SmolVLAPolicy":
        """Load a SmolVLA policy from a local directory or HF Hub repo id.

        Args:
            pretrained_model_name_or_path: local path or HF Hub repo id (e.g. "lerobot/smolvla_base").
            config_overrides: dict of SmolVLAConfig fields to override after loading.
            strict: forward to nn.Module.load_state_dict — set False if you expect key drift
                (e.g. different action_dim projection shapes after fine-tuning).
        """
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            path = Path(snapshot_download(str(pretrained_model_name_or_path)))

        with open(path / "config.json") as f:
            config_data = json.load(f)
        config = SmolVLAConfig.from_dict(config_data)
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        policy = cls(config)
        state_dict = load_file(path / "model.safetensors", device=str(map_location) if map_location else "cpu")
        policy.load_state_dict(state_dict, strict=strict)
        return policy
