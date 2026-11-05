#!/usr/bin/env bash
# Create the env for training and for running on the robot, plus the two repos the code needs.
#   bash env/setup.sh                 # conda env `lerobot` (name used throughout the docs)
#   ENV_PREFIX=/path bash env/setup.sh  # or a conda env at a path
# Needs conda/mamba (e.g. Miniforge) and git. NVIDIA driver must support CUDA 12.8 (driver >= 570).
#   ~/norma-core       NormaCore at the Station release used at the hackathon (station_py, protobuf stubs)
#   ~/robot-pizza-cup  organizers' repo (Station parquet -> LeRobot v2.1 exporter, used by pipeline.sh prep)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
ENV_NAME=${ENV_NAME:-lerobot}
STATION_VERSION=${STATION_VERSION:-v0.1.0-beta.10}
NORMA_CORE_ROOT=${NORMA_CORE_ROOT:-$HOME/norma-core}
if [ -n "${ENV_PREFIX:-}" ]; then WHERE=(-p "$ENV_PREFIX"); else WHERE=(-n "$ENV_NAME"); fi

conda create -y "${WHERE[@]}" python=3.10
conda run "${WHERE[@]}" --no-capture-output pip install -r "$REPO/env/requirements.lock" \
  --extra-index-url https://download.pytorch.org/whl/cu128

if [ -d "$NORMA_CORE_ROOT" ]; then
  echo "norma-core: using existing $NORMA_CORE_ROOT ($(git -C "$NORMA_CORE_ROOT" describe --tags --always 2>/dev/null || echo '?'))"
else
  git clone --depth 1 --branch "$STATION_VERSION" https://github.com/norma-core/norma-core.git "$NORMA_CORE_ROOT"
fi
[ -d ~/robot-pizza-cup ] || git clone --depth 1 https://github.com/FunRobots/robot-pizza-cup.git ~/robot-pizza-cup

conda run "${WHERE[@]}" --no-capture-output python "$REPO/env/check_env.py" --norma-root "$NORMA_CORE_ROOT"
echo "activate: conda activate ${ENV_PREFIX:-$ENV_NAME}"
