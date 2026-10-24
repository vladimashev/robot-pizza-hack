#!/usr/bin/env bash
# Station parquet (with quadrant prompts) -> LeRobot v2.1 dataset, using the organizers' exporter.
# Cameras: cam0=scene -> "top", cam1=wrist -> "front" (naming required by the hackathon guide).
set -euo pipefail
SRC=${1:-$HOME/quadrant_src}
OUT=${2:-$HOME/lerobot_cheese}
[ -d ~/robot-pizza-cup ] || git clone https://github.com/FunRobots/robot-pizza-cup.git ~/robot-pizza-cup
python ~/robot-pizza-cup/elrobot/scripts/export_to_lerobot.py "$SRC" "$OUT" --camera-names top front
cat "$OUT/meta/tasks.jsonl"
