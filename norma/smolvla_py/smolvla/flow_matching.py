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

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from smolvla.config import SmolVLAConfig
from smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from smolvla.utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
)


class VLAFlowMatching(nn.Module):
    """SmolVLA flow-matching head over a frozen (or partially trained) VLM + action expert."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            load_vlm_weights=config.load_vlm_weights,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            num_vlm_layers=config.num_vlm_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
            device=config.device if config.device is not None else "auto",
        )

        vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
        expert_hidden = self.vlm_with_expert.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, vlm_hidden)
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        self.set_requires_grad()

        tokenizer = self.vlm_with_expert.processor.tokenizer
        self.fake_image_token = tokenizer.fake_image_token_id
        self.global_image_token = tokenizer.global_image_token_id
        self.register_buffer(
            "global_image_start_token",
            torch.tensor([self.fake_image_token, self.global_image_token], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "image_end_token",
            torch.tensor([self.fake_image_token], dtype=torch.long),
            persistent=False,
        )

        self.add_image_special_tokens = config.add_image_special_tokens
        self.prefix_length = config.prefix_length

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        return time_beta * 0.999 + 0.001

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images + language + state into a single prefix sequence for the VLM stream."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                start_emb = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                start_mask = torch.ones_like(start_emb[:, :, 0], dtype=torch.bool, device=start_emb.device)
                att_masks += [0] * start_mask.shape[-1]
                embs.append(start_emb)
                pad_masks.append(start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                end_emb = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                end_mask = torch.ones_like(end_emb[:, :, 0], dtype=torch.bool, device=end_emb.device)
                embs.append(end_emb)
                pad_masks.append(end_mask)
                att_masks += [0] * end_mask.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)

        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy actions + timestep for the expert stream."""
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        att_masks = torch.tensor(
            [1] * self.config.chunk_size, dtype=action_time_emb.dtype, device=device
        )[None, :].expand(bsize, -1)

        return action_time_emb, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Training forward — returns per-element losses of shape (B, chunk_size, max_action_dim)."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Inference forward — iterative denoising, returns (B, chunk_size, max_action_dim)."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """One denoising step; expert cross-attends into the cached prefix KV."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
