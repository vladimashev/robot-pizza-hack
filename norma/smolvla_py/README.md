# 🤖 smolvla_py

SmolVLA fine-tune for the SO-101 arm. Self-contained Python port — trains on
norma-core parquet datasets, deploys against the station hardware.

Built on [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) and
[`lerobot`](https://github.com/huggingface/lerobot) by 🤗 HuggingFace.
See [Credits](#-credits) below.

## ☁️ Train remotely

For a single serverless fine-tune run (e.g. on Nebius GPUs), bundle the code
and dataset into a self-contained zip and ship it to the remote.

**1. Build the bundle:**

```bash
./scripts/build_bundle.sh path/to/dataset.parquet [more.parquet ...]
```

Produces `smolvla-bundle.zip` with `pyproject.toml`, `uv.lock`, the `smolvla/`
package, `scripts/train.py`, `scripts/merge_datasets.py`, and the staged
parquets under `datasets/`. Everything the trainer needs is inside.

**2. Submit the job.** Bundling is the only step that differs from a local
run; everything else — `uv sync`, `scripts/train.py`, hyperparam flags — is
identical, just executed on the remote against the bundle's own `datasets/`
directory. 

The full Nebius recipe (upload bundle, create bucket, create job,
fetch checkpoints) is in [`nebius.md`](./nebius.md).

## 🏋️ Train locally

Same training entry point, run directly on your own GPU — useful for short
development runs or if you'd rather not spin up a cloud job:

```bash
uv sync
uv run python scripts/train.py \
    --steps 30000 --batch-size 64 \
    --parquets path/to/dataset.parquet \
    --output checkpoints/run
```

Pass multiple `--parquets` to concat episodes across files. Stats are computed
over the full set. Checkpoints land in `--output/step-NNNNNN/` and
`--output/final/`. See `--help` for the full hyperparam list (`--lr`,
`--warmup-steps`, `--decay-steps`, `--save-every`, ...).

If you have many parquets to merge first:

```bash
uv run python scripts/merge_datasets.py \
    --inputs path/to/dataset-*.parquet \
    --output path/to/merged.parquet
```

## 🦾 Evaluate on device

There is no offline eval — validation happens on the real arm. Pre-reqs:

- the station daemon is running on the machine wired to the SO-101 over USB.
  `--server` is that machine's hostname (default: `localhost`). For our setup
  the daemon runs on a Pi 5 reachable as `ab-rpi5.server`; pass that via
  `--server` if you're driving from a different box.
- you know the **ST3215 bus id** from the station web interface (open the
  station viewer, look at the bus card). Pass it as `--bus-serial`.

Interactive — SPACE per tick, `q` / Ctrl-C to quit:

```bash
uv run python scripts/run_policy.py \
    --checkpoint     checkpoints/run/final \
    --task           "place the yellow cube on top of the other" \
    --bus-serial     5AB9068807 \
    --max-delta-ticks 512
```

Continuous — `--auto` runs predict+send in a loop:

```bash
uv run python scripts/run_policy.py \
    --checkpoint  checkpoints/run/final \
    --task        "place the yellow cube on top of the other" \
    --bus-serial  5AB9068807 \
    --auto --max-ticks 200
```

Each tick fetches a frame from `inference/normvla`, predicts an action chunk,
sends a clipped goal, then prints the resulting position delta.

## 🙏 Credits

This is a minimal port; the model architecture, base checkpoint, and most of
the training-side code are from 🤗 HuggingFace:

- **SmolVLA** — vision-language-action model. Docs:
  <https://huggingface.co/docs/lerobot/smolvla>. Base checkpoint
  `lerobot/smolvla_base` is what we fine-tune from.
- **`lerobot`** — <https://github.com/huggingface/lerobot>. Files in
  `smolvla/` carrying the `Copyright 2025 HuggingFace Inc. team` header are
  direct ports / adaptations of code from lerobot. Both upstream and our
  modifications are Apache-2.0.

Norma-core-specific additions (parquet ingest, on-device deployment loop,
remote-job tooling) carry `Copyright 2026 Norma Core contributors` and are
also Apache-2.0.
