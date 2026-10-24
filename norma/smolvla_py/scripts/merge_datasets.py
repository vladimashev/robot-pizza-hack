"""Merge several norma-core parquet files into one.

Streams batches via pyarrow.ParquetWriter — never loads all files into RAM,
so it handles the big dataset-5 without issue. Schema check up front.

Run:
    uv run python scripts/merge_datasets.py \\
        --inputs ../datasets/dataset-*.parquet \\
        --output ../datasets/dataset-merged.parquet
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input parquet paths (shell globs OK).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compression", default="snappy",
                   choices=["snappy", "zstd", "gzip", "none"])
    return p.parse_args()


def expand_globs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches:
            raise SystemExit(f"no files match: {pat}")
        files.extend(Path(m) for m in matches)
    return files


def main() -> None:
    args = parse_args()
    inputs = expand_globs(args.inputs)

    print(f"merging {len(inputs)} files:")
    for p in inputs:
        print(f"  {p}")

    # Validate schemas match up front.
    schemas = [pq.read_schema(p) for p in inputs]
    base = schemas[0]
    for path, sch in zip(inputs[1:], schemas[1:]):
        if sch != base:
            raise SystemExit(f"schema mismatch between {inputs[0]} and {path}")
    print(f"schema: {base.names}")

    total_rows = sum(pq.read_metadata(p).num_rows for p in inputs)
    print(f"total rows: {total_rows}")

    compression = None if args.compression == "none" else args.compression
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with pq.ParquetWriter(args.output, base, compression=compression) as writer:
        for p in inputs:
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=1024):
                writer.write_batch(batch)
                written += batch.num_rows
                if written % 10000 < 1024:
                    print(f"  {written}/{total_rows} rows ...")
    print(f"wrote {written} rows to {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
