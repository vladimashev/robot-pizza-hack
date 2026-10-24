# Copyright 2026 Norma Core contributors.
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

"""Compute and persist (state, action) mean/std for SmolVLA-style normalization.

Persistence format is a single safetensors file with four tensors:

    state_mean   (state_dim,)  float32
    state_std    (state_dim,)  float32
    action_mean  (action_dim,) float32
    action_std   (action_dim,) float32

Small std floor (`eps`) is applied to avoid divide-by-zero on joints that
barely move in training (common for wrist_roll on pick-place tasks).
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def compute_stats(
    state: torch.Tensor | "np.ndarray",  # noqa: F821 — np imported lazily only if needed
    action: torch.Tensor | "np.ndarray",  # noqa: F821
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Compute per-feature mean/std over all training frames.

    `state` and `action` are flat (N, D) arrays — no episode structure needed.
    """
    state_t = torch.as_tensor(state, dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.float32)
    if state_t.ndim != 2 or action_t.ndim != 2:
        raise ValueError(
            f"state and action must be 2D (N, D); got {state_t.shape}, {action_t.shape}"
        )
    return {
        "state_mean": state_t.mean(dim=0),
        "state_std": state_t.std(dim=0).clamp_min(eps),
        "action_mean": action_t.mean(dim=0),
        "action_std": action_t.std(dim=0).clamp_min(eps),
    }


def save_stats(stats: dict[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in stats.items()}, str(path))


def load_stats(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))
