# Single fine-tune run on Nebius serverless API

Pre-reqs: `nebius` CLI logged in, `s5cmd` with a `nebius` profile in
`~/.aws/credentials`.

## 1. Create bucket

```bash
nebius storage bucket create --name normacore-so101-sample
nebius storage bucket list                              # grab the bucket id
```

Note the resource id (`storagebucket-...`) — you reference the bucket by id
in the `--volume` flag, which lets the job auth via project IAM (no AWS creds
needed).

The S3 endpoint for `s5cmd` follows the bucket's region:
`https://storage.<region>.nebius.cloud` (e.g. `eu-north1`).

## 2. Build bundle and upload

The bundle is a self-contained zip with everything the job needs to train:
`pyproject.toml`, `uv.lock`, the `smolvla/` package, the `scripts/` directory,
and one or more parquet datasets staged under `datasets/`. On the remote, the
job extracts it, runs `uv sync` against the lockfile, and trains.

Build:

```bash
./scripts/build_bundle.sh ../datasets/dataset.parquet                     # single parquet
./scripts/build_bundle.sh ../datasets/a.parquet ../datasets/b.parquet     # merged
```

Output: `smolvla-bundle.zip` next to `pyproject.toml`. Re-run any time code
or datasets change — the script is idempotent and overwrites the previous zip.

Upload:

```bash
s5cmd --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud \
    cp smolvla-bundle.zip s3://normacore-so101-sample/code/smolvla-bundle.zip
```

## 3. Run job

Docs:
- Jobs API: <https://docs.nebius.com/serverless/quickstart/jobs>
- `--platform` / `--preset` reference: <https://docs.nebius.com/compute/virtual-machines/types>

```bash
RUN_TAG=fine-tune-$(date +%F-%H%M)

nebius ai job create \
  --name      smolvla-finetune-$(date +%s) \
  --image     ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  --platform  gpu-l40s-d \
  --preset    1gpu-16vcpu-96gb \
  --timeout   6h \
  --volume    storagebucket-<your-bucket-id>:/mnt/bucket:rw \
  --env       STEPS=10000 \
  --env       BATCH_SIZE=64 \
  --env       RUN_TAG=$RUN_TAG \
  --working-dir       /workspace \
  --container-command /bin/bash \
  --args '-lc "python -m zipfile -e /mnt/bucket/code/smolvla-bundle.zip . && cd smolvla-bundle && uv sync && uv run python scripts/train.py --steps $STEPS --batch-size $BATCH_SIZE --parquets datasets/dataset.parquet --output /mnt/bucket/runs/$RUN_TAG"'
```

Watch:
```bash
nebius ai job list
nebius ai job logs <job-id> --follow
```

## 4. Download results

Pick the step (or `final`) you want and pull just that checkpoint:

```bash
s5cmd --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud \
    cp "s3://normacore-so101-sample/runs/fine-tune-2026-04-30-0042/step-014000/*" ./
```

List what's available first:

```bash
s5cmd --profile nebius --endpoint-url https://storage.eu-north1.nebius.cloud \
    ls "s3://normacore-so101-sample/runs/$RUN_TAG/"
```
