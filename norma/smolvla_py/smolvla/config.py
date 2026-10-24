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

from dataclasses import dataclass, field, fields


@dataclass
class SmolVLAConfig:
    # I/O shapes
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_dim: int = 32  # real (unpadded) action dim — used to slice outputs
    state_dim: int = 32   # real (unpadded) state dim — currently informational only
    image_keys: list[str] = field(default_factory=list)

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 48
    pad_language_to: str = "longest"  # or "max_length"

    # Flow-matching
    num_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0

    # Attention / prefix
    use_cache: bool = True
    add_image_special_tokens: bool = False
    prefix_length: int = -1

    # Finetuning toggles
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # VLM / expert architecture
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = False
    attention_mode: str = "cross_attn"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    device: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SmolVLAConfig":
        """Build a config from a dict, silently ignoring unknown keys.

        Lets us consume lerobot-flavored config.json files that carry extra
        framework fields (feature specs, optimizer presets, RTC, etc.).
        """
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
