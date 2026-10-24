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

"""Apply / invert the (x - mean) / std transform paired with smolvla.stats.

The stats dict is the one produced by `smolvla.stats.compute_stats`.
"""

from __future__ import annotations

from torch import Tensor


def _bcast(stat: Tensor, x: Tensor) -> Tensor:
    """Reshape (D,) stat so it broadcasts against x of shape (..., D)."""
    while stat.ndim < x.ndim:
        stat = stat.unsqueeze(0)
    return stat.to(device=x.device, dtype=x.dtype)


def normalize_state(x: Tensor, stats: dict[str, Tensor]) -> Tensor:
    return (x - _bcast(stats["state_mean"], x)) / _bcast(stats["state_std"], x)


def normalize_action(x: Tensor, stats: dict[str, Tensor]) -> Tensor:
    return (x - _bcast(stats["action_mean"], x)) / _bcast(stats["action_std"], x)


def unnormalize_action(x: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Inverse of normalize_action — run on policy outputs at inference."""
    return x * _bcast(stats["action_std"], x) + _bcast(stats["action_mean"], x)
