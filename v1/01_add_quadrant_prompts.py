"""Rewrite the `task` column of a NormaCore Station parquet with per-quadrant prompts.

Episode -> quadrant mapping comes from the manifest (`ep` 1..4 = TL, TR, BR, BL,
fixed clockwise order). Output goes to a separate folder; the original is untouched.

Usage: python 01_add_quadrant_prompts.py ~/datasets/cheese-fixed-s1-2026-10-24 ~/quadrant_src
       (first arg = path prefix of <name>.parquet / <name>.manifest.json)
"""
import json, os, sys
import pyarrow as pa, pyarrow.parquet as pq

QUADRANTS = {1: "top-left", 2: "top-right", 3: "bottom-right", 4: "bottom-left"}
PROMPT = "put the cheese on the {} quadrant of the pizza"

src = os.path.expanduser(sys.argv[1])
out_dir = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/quadrant_src")

manifest = json.load(open(f"{src}.manifest.json"))
ep2task = {e["episode_start_ns"]: PROMPT.format(QUADRANTS[e["ep"]]) for e in manifest["episodes"]}

t = pq.read_table(f"{src}.parquet")
ids = t["episode_start_ns"].to_pylist()
missing = set(ids) - ep2task.keys()
assert not missing, f"episodes missing from manifest: {missing}"
t = t.set_column(t.schema.get_field_index("task"), "task", pa.array([ep2task[i] for i in ids], pa.string()))

os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, os.path.basename(src) + "-quadrants.parquet")
pq.write_table(t, out, compression="zstd")
print(t.group_by("task").aggregate([("episode_start_ns", "count_distinct")]))
print("wrote", out)
