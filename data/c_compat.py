"""Merge ~/v2/lerobot/<src> (read-only) into lerobot-0.3.3 training sets in ~/v2/lerobot_compat/<variant>/.
- trims idle frames at episode start/end (arm waiting at home)
- variants: full / nogripper (state dim 7 zeroed) / nostate (all state zeroed)
- fixes datasets>=4 "List" metadata, renumbers episodes 0..N-1, merges task lists
Videos are symlinked, not copied."""
import json, math, os, shutil, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

H = os.path.expanduser("~")
SRC_ROOT, OUT_ROOT = f"{H}/v2/lerobot", f"{H}/v2/lerobot_compat"
assert not os.path.realpath(OUT_ROOT).startswith(os.path.realpath(f"{H}/datasets"))
SOURCES = ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24", "pepperoni-on-cheese-r1r5-2026-10-24"]
VARIANTS = {"full": [], "nogripper": [7], "nostate": "all"}
THR, KEEP, MIN_FRAC = 0.01, 5, 0.5   # idle threshold (norm units), frames kept before/after motion, max trim


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def setc(t, name, arr):
    return t.set_column(t.schema.get_field_index(name), name, arr)


def fix_meta(t):
    md = dict(t.schema.metadata or {})
    if b"huggingface" in md:
        md[b"huggingface"] = json.dumps(json.loads(md[b"huggingface"])).replace(
            '"_type": "List"', '"_type": "Sequence"').encode()
    return md


def trim_bounds(t):
    s = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
    a = np.array(t["action"].to_pylist(), dtype=np.float32)
    moving = np.abs(a - s).max(1) > THR
    moving[1:] |= np.abs(np.diff(a, axis=0)).max(1) > THR / 2
    idx, n = np.flatnonzero(moving), len(s)
    if not len(idx):
        return 0, n
    lo, hi = max(0, idx[0] - KEEP), min(n, idx[-1] + 1 + KEEP)
    return (lo, hi) if hi - lo >= MIN_FRAC * n else (0, n)


info0, eps, tasks = None, [], {}
for src in SOURCES:
    d = f"{SRC_ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    shapes = {k: v["shape"] for k, v in info["features"].items()}
    if info0 is None:
        info0, shapes0 = info, shapes
    elif shapes != shapes0 or info["fps"] != info0["fps"]:
        sys.exit(f"{src}: features/fps differ from {SOURCES[0]}:\n{shapes}\nvs\n{shapes0}")
    tmap = {x["task_index"]: x["task"] for x in jl(f"{d}/meta/tasks.jsonl")}
    stats = {x["episode_index"]: x for x in jl(f"{d}/meta/episodes_stats.jsonl")}
    before = after = 0
    for e in jl(f"{d}/meta/episodes.jsonl"):
        i = e["episode_index"]
        ch = i // info["chunks_size"]
        dp = f"{d}/" + info["data_path"].format(episode_chunk=ch, episode_index=i)
        lo, hi = trim_bounds(pq.read_table(dp, columns=["observation.state", "action"]))
        for tk in e["tasks"]:
            tasks.setdefault(tk, len(tasks))
        eps.append(dict(d=d, info=info, old=i, chunk=ch, dp=dp, lo=lo, hi=hi,
                        tmap=tmap, stats=stats[i], tasks=e["tasks"]))
        before += e["length"]; after += hi - lo
    print(f"{src}: {before} -> {after} frames after trimming ({100 * (1 - after / before):.0f}% idle removed)")

cams = [k for k, v in info0["features"].items() if v["dtype"] == "video"]
sdim = info0["features"]["observation.state"]["shape"][0]
cs = info0["chunks_size"]
print("tasks:", *tasks, sep="\n  ")

for var, zero in VARIANTS.items():
    zd = list(range(sdim)) if zero == "all" else zero
    dst = f"{OUT_ROOT}/{var}"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/meta")
    new_eps, new_stats, off = [], [], 0
    for new, e in enumerate(eps):
        t = pq.read_table(e["dp"])
        md = fix_meta(t)
        t = t.slice(e["lo"], e["hi"] - e["lo"])
        n = t.num_rows
        t = setc(t, "episode_index", pa.array([new] * n, pa.int64()))
        t = setc(t, "index", pa.array(range(off, off + n), pa.int64()))
        t = setc(t, "task_index", pa.array([tasks[e["tmap"][k]] for k in t["task_index"].to_pylist()], pa.int64()))
        if zd:
            st = np.array(t["observation.state"].to_pylist(), dtype=np.float32)
            st[:, zd] = 0.0
            t = setc(t, "observation.state", pa.array(st.tolist(), t.schema.field("observation.state").type))
        p = f"{dst}/" + info0["data_path"].format(episode_chunk=new // cs, episode_index=new)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pq.write_table(t.replace_schema_metadata(md), p)
        for c in cams:
            sp = f"{e['d']}/" + e["info"]["video_path"].format(episode_chunk=e["chunk"], video_key=c, episode_index=e["old"])
            vp = f"{dst}/" + info0["video_path"].format(episode_chunk=new // cs, video_key=c, episode_index=new)
            os.makedirs(os.path.dirname(vp), exist_ok=True)
            os.symlink(sp, vp)
        s = json.loads(json.dumps(e["stats"]))
        s["episode_index"] = new
        for k in ("min", "max", "mean", "std"):
            for dd in zd:
                s["stats"]["observation.state"][k][dd] = 0.0
        new_stats.append(s)
        new_eps.append({"episode_index": new, "tasks": e["tasks"], "length": n})
        off += n
    ne = len(new_eps)
    json.dump({**info0, "total_episodes": ne, "total_frames": off, "total_tasks": len(tasks),
               "total_videos": ne * len(cams), "total_chunks": math.ceil(ne / cs),
               "splits": {"train": f"0:{ne}"}}, open(f"{dst}/meta/info.json", "w"), indent=2)
    open(f"{dst}/meta/tasks.jsonl", "w").writelines(
        json.dumps({"task_index": i, "task": tk}) + "\n" for tk, i in sorted(tasks.items(), key=lambda x: x[1]))
    open(f"{dst}/meta/episodes.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_eps)
    open(f"{dst}/meta/episodes_stats.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_stats)
    json.dump({"zero_state_dims": zd, "sources": SOURCES, "trim": {"thr": THR, "keep": KEEP}},
              open(f"{dst}/meta/v2_notes.json", "w"), indent=2)
    print(f"{dst}: {ne} episodes, {off} frames, zeroed state dims: {zd or 'none'}")
