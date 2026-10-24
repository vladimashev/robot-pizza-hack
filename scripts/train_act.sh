#!/usr/bin/env bash
# Four ACT policies in parallel, one per quadrant (ACT ignores the prompt).
# Env: conda env `lerobot` with lerobot==0.3.3 and torch 2.7.1+cu128.
set -euo pipefail
STEPS=${STEPS:-30000}
for Q in top-left top-right bottom-right bottom-left; do
  rm -rf ~/outputs/act_$Q
  nohup python -m lerobot.scripts.train --policy.type=act \
    --dataset.repo_id=local/act_$Q --dataset.root=$HOME/act_data/$Q \
    --batch_size=8 --steps=$STEPS --save_freq=10000 \
    --output_dir=$HOME/outputs/act_$Q --job_name=act_$Q \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
    "$@" > ~/train_act_$Q.log 2>&1 &
done
echo "progress: grep step: ~/train_act_*.log | tail"
