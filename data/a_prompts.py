"""Station parquets from ~/datasets (read-only) -> ~/v2/station/<name>/ with the manifest's per-episode task."""
import collections, json, os, sys
import pyarrow as pa, pyarrow.parquet as pq

H = os.path.expanduser("~")
SRC, OUT = f"{H}/datasets", f"{H}/v2/station"
assert not os.path.realpath(OUT).startswith(os.path.realpath(SRC)), "refusing to write into ~/datasets"
SOURCES = ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24", "pepperoni-on-cheese-r1r5-2026-10-24"]

for name in SOURCES:
    m = json.load(open(f"{SRC}/{name}.manifest.json"))
    eps = m["episodes"]
    if not all(e.get("task") and "episode_start_ns" in e for e in eps):
        sys.exit(f"{name}: some episodes lack task/episode_start_ns")
    ep2task = {e["episode_start_ns"]: e["task"].strip() for e in eps}
    t = pq.read_table(f"{SRC}/{name}.parquet")
    ids = t["episode_start_ns"].to_pylist()
    missing = set(ids) - ep2task.keys()
    if missing:
        sys.exit(f"{name}: {len(missing)} episodes in parquet not in manifest")
    old = collections.Counter(t["task"].to_pylist())
    t = t.set_column(t.schema.get_field_index("task"), "task", pa.array([ep2task[i] for i in ids], pa.string()))
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    pq.write_table(t, f"{OUT}/{name}/{name}.parquet", compression="zstd")
    print(f"\n=== {name}: {len(eps)} eps, targets {dict(collections.Counter(e['target'] for e in eps))}")
    print(f"  parquet task column had {len(old)} distinct value(s) before")
    for k in ("notes", "tasks_note", "missing_episodes"):
        if m.get(k):
            print(f"  {k}: {m[k]}")
