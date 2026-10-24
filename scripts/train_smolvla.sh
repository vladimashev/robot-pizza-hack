#!/usr/bin/env bash
# One SmolVLA for all quadrants; the prompt in the `task` column selects the quadrant.
# Env: conda env `lerobot` with lerobot[smolvla]==0.3.3 and torch 2.7.1+cu128.
set -euo pipefail
STEPS=${STEPS:-20000}
OUT=${OUT:-$HOME/outputs/smolvla_quadrants}
rm -rf "$OUT"
nohup python -m lerobot.scripts.train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/lerobot_cheese_act --dataset.root=$HOME/lerobot_cheese_act \
  --dataset.image_transforms.enable=true \
  --batch_size=64 --steps=$STEPS --save_freq=5000 \
  --output_dir="$OUT" --job_name=$(basename "$OUT") \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  "$@" > ~/train_$(basename "$OUT").log 2>&1 &
echo "progress: grep step: ~/train_$(basename "$OUT").log | tail -3"
