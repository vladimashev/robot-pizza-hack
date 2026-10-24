# Cheese-on-pizza: per-quadrant policies for ElRobot

Pipeline used at the Robotics Pizza Hackathon (Oct 2026).

## Data
Station parquet + manifest. One episode = one quadrant, one round = 4 episodes in fixed
clockwise order: `ep` 1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left.

Prompt template (must match exactly at deploy):
`put the cheese on the {top-left|top-right|bottom-right|bottom-left} quadrant of the pizza`

## Steps
    python scripts/01_add_quadrant_prompts.py ~/datasets/cheese-fixed-s1-2026-10-24 ~/quadrant_src
    bash   scripts/02_convert_to_lerobot.sh ~/quadrant_src ~/lerobot_cheese   # base env (datasets>=4 ok)
    conda activate lerobot                                                    # lerobot==0.3.3, torch cu128
    python scripts/03_make_compat_datasets.py
    bash   scripts/train_act.sh        # 4 ACT models -> ~/outputs/act_<quadrant>
    bash   scripts/train_smolvla.sh    # 1 SmolVLA    -> ~/outputs/smolvla_quadrants

## Deploy
`scripts/run_lerobot_policy.py` runs a LeRobot checkpoint (ACT or SmolVLA) against a live
NormaCore Station. Place it next to norma-core `smolvla_py/scripts/run_policy.py` (or pass
`--norma-root`). Start with `--dry-run`. ACT: one checkpoint per quadrant, no prompt,
use `--n-action-steps 10` (data is 10 Hz).

## Notes
- `norma/smolvla_py` is the NormaCore trainer; its own `run_policy.py` only loads
  smolvla_py checkpoints and drives 6 motors by default.
- Round 3 episodes are flagged `cameras_swapped` in the manifest; verify visually.
