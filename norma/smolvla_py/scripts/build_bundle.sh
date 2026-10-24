#!/usr/bin/env bash
# Build a self-contained zip for remote fine-tuning (Nebius, vast.ai, etc.).
#
# Stays scoped to the smolvla/ folder. Includes:
#
#     pyproject.toml
#     uv.lock        (for reproducible `uv sync` on the remote)
#     smolvla/       (Python package)
#     scripts/       (CLI entry points; run_policy.py and build_bundle.sh
#                     are excluded — run_policy needs the norma-core monorepo
#                     to import station_py + protobufs, and the remote
#                     doesn't bundle anything itself)
#     datasets/      (populated from paths you pass on the CLI)
#
# So on the remote:
#
#     unzip smolvla-bundle.zip
#     cd smolvla-bundle
#     uv sync
#     uv run python scripts/train.py --parquets datasets/dataset.parquet
#
# Validation happens on device, not in this loop.
#
# Parquets are already snappy-compressed — `zip -0` (store, no deflate) saves
# time with basically the same output size.
#
# Usage:
#   ./scripts/build_bundle.sh <dataset1.parquet> [<dataset2.parquet> ...]
#
# Example:
#   ./scripts/build_bundle.sh ../datasets/dataset.parquet

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <dataset1.parquet> [<dataset2.parquet> ...]" >&2
    echo "(no datasets passed — cowardly refusing to build a code-only bundle)" >&2
    exit 2
fi

OUT="$HERE/smolvla-bundle.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/smolvla-bundle/datasets"

rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
      --exclude 'run_policy.py' --exclude 'build_bundle.sh' \
      pyproject.toml uv.lock smolvla scripts "$STAGE/smolvla-bundle/"

for src in "$@"; do
    if [[ ! -f "$src" ]]; then
        echo "missing: $src" >&2; exit 1
    fi
    cp "$src" "$STAGE/smolvla-bundle/datasets/$(basename "$src")"
    echo "  + datasets/$(basename "$src")"
done

rm -f "$OUT"
( cd "$STAGE" && zip -r -0 "$OUT" smolvla-bundle >/dev/null )

echo
echo "done:  $(du -h "$OUT" | cut -f1)  →  $OUT"
