#!/usr/bin/env bash
# v2 pipeline end to end, in stages:  bash pipeline.sh [prep] [policy] [checkers]   (default: all three)
#   prep      ~/datasets -> prompts from manifests -> LeRobot v2.1 (~/v2/lerobot) -> merged training sets
#   policy    one SmolVLA on ~/v2/lerobot_compat/$VARIANT -> ~/outputs/smolvla_v2_$VARIANT
#   checkers  roi.json, topping checker (checker.pt), change checker (change.pt)
# Inputs: ~/datasets/<src>.parquet + <src>.manifest.json (see a_prompts.py SOURCES).
# Env: conda env `lerobot` with lerobot[smolvla]==0.3.3, torch cu128, pyav, opencv, pandas.
# VARIANT=full|nogripper|nostate (c_compat.py). nogripper/nostate zero state dims at train time, so the robot
# runner must zero the same dims (run_challenge.py --zero-state-dims). Which variant was deployed is not recorded.
set -euo pipefail
cd "$(dirname "$0")"
STAGES=${*:-prep policy checkers}
VARIANT=${VARIANT:-full}
STEPS=${STEPS:-20000}
OUT=${OUT:-$HOME/outputs/smolvla_v2_$VARIANT}
SOURCES="cheese-r1r10-2026-10-24 pepperoni-r1r10-2026-10-24 pepperoni-on-cheese-r1r5-2026-10-24"

for stage in $STAGES; do case $stage in
prep)
  python a_prompts.py                                         # -> ~/v2/station/<src>/<src>.parquet
  [ -d ~/robot-pizza-cup ] || git clone https://github.com/FunRobots/robot-pizza-cup.git ~/robot-pizza-cup
  for s in $SOURCES; do                                       # exporter refuses existing outputs
    [ -d ~/v2/lerobot/$s ] || python ~/robot-pizza-cup/elrobot/scripts/export_to_lerobot.py \
      ~/v2/station/$s ~/v2/lerobot/$s --camera-names top front
  done
  python analyze.py                                           # -> ~/v2/analysis, review flagged.csv
  python c_compat.py                                          # -> ~/v2/lerobot_compat/{full,nogripper,nostate}
  ;;
policy)
  rm -rf "$OUT"
  python -m lerobot.scripts.train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/v2_$VARIANT --dataset.root=$HOME/v2/lerobot_compat/$VARIANT \
    --dataset.image_transforms.enable=true \
    --batch_size=64 --steps=$STEPS --save_freq=5000 \
    --output_dir="$OUT" --job_name=$(basename "$OUT") \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
    2>&1 | tee ~/train_$(basename "$OUT").log
  ;;
checkers)
  python d_roi.py                                             # -> ~/v2/checker/roi.json (check: pizza [118,131,28])
  python e_checker_dataset.py && python f_train_checker.py    # -> checker.pt (topping present?)
  python g_pair_dataset.py && python h_train_change.py        # -> change.pt (topping added?)
  python failure_checker_v2.py                                # smoke test on the reference episode
  ;;
*) echo "unknown stage: $stage (prep|policy|checkers)"; exit 1 ;;
esac; done
