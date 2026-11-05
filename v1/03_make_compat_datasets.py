"""Build lerobot-0.3.3-compatible copies of the converted dataset. Original stays untouched.

Fixes two issues:
  1. Parquets written by `datasets` >= 4 store feature type "List"; lerobot 0.3.3 ships
     `datasets` 3.6 which only knows "Sequence". Metadata is patched in the copies.
  2. lerobot 0.3.x breaks on --dataset.episodes with non-contiguous indices, so per-quadrant
     subsets are built as standalone datasets with episodes renumbered 0..N-1.

Outputs (videos are symlinked, not copied):
  ~/lerobot_cheese_act/        all episodes, 4 prompts   -> SmolVLA
  ~/act_data/<quadrant>/       one quadrant each         -> ACT (one model per quadrant)
"""
import json, os, shutil
import pyarrow as pa, pyarrow.parquet as pq

H = os.path.expanduser("~")
SRC = f"{H}/lerobot_cheese"
QUADRANTS = ["top-left", "top-right", "bottom-right", "bottom-left"]

info = json.load(open(f"{SRC}/meta/info.json"))
eps = [json.loads(l) for l in open(f"{SRC}/meta/episodes.jsonl")]
stats = {s["episode_index"]: s for s in map(json.loads, open(f"{SRC}/meta/episodes_stats.jsonl"))}
cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]


def fix_meta(t):
    md = dict(t.schema.metadata or {})
    if b"huggingface" in md:
        md[b"huggingface"] = json.dumps(json.loads(md[b"huggingface"])).replace(
            '"_type": "List"', '"_type": "Sequence"').encode()
    return md


def build(dst, selected):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/meta")
    os.makedirs(f"{dst}/data/chunk-000")
    new_eps, new_stats, off = [], [], 0
    for new, e in enumerate(selected):
        old = e["episode_index"]
        t = pq.read_table(f"{SRC}/data/chunk-000/episode_{old:06d}.parquet")
        md, n = fix_meta(t), t.num_rows
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array([new] * n, pa.int64()))
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(range(off, off + n), pa.int64()))
        pq.write_table(t.replace_schema_metadata(md), f"{dst}/data/chunk-000/episode_{new:06d}.parquet")
        for c in cams:
            os.makedirs(f"{dst}/videos/chunk-000/{c}", exist_ok=True)
            os.symlink(f"{SRC}/videos/chunk-000/{c}/episode_{old:06d}.mp4",
                       f"{dst}/videos/chunk-000/{c}/episode_{new:06d}.mp4")
        new_eps.append({**e, "episode_index": new})
        new_stats.append({**stats[old], "episode_index": new})
        off += n
    json.dump({**info, "total_episodes": len(selected), "total_frames": off,
               "total_videos": len(selected) * len(cams), "total_chunks": 1,
               "splits": {"train": f"0:{len(selected)}"}}, open(f"{dst}/meta/info.json", "w"), indent=2)
    shutil.copy(f"{SRC}/meta/tasks.jsonl", f"{dst}/meta/")
    open(f"{dst}/meta/episodes.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_eps)
    open(f"{dst}/meta/episodes_stats.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_stats)
    print(f"{dst}: {len(selected)} episodes, {off} frames")


build(f"{H}/lerobot_cheese_act", eps)
for q in QUADRANTS:
    build(f"{H}/act_data/{q}", [e for e in eps if q in e["tasks"][0]])
