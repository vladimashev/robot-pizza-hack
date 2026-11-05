"""Remove idle frames (start/end + mid-episode pauses) from converted LeRobot v2.1 datasets.
Reads ~/v2/lerobot/<src> (read-only). Writes ~/v2/lerobot_clean/<src>: data with contiguous
timestamps, re-encoded videos, recomputed state/action stats, meta/clean_report.json.
Motion = any change in action OR state on any of the 8 joints (gripper included),
so grasp moments (arm still, gripper closing) are never cut."""
import json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, av

H = os.path.expanduser("~")
SRC_ROOT, OUT_ROOT = f"{H}/v2/lerobot", f"{H}/v2/lerobot_clean"
assert not os.path.realpath(OUT_ROOT).startswith(os.path.realpath(f"{H}/datasets"))
SOURCES = sys.argv[1:] or ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24",
                           "pepperoni-on-cheese-r1r5-2026-10-24"]
IDLE = 0.005      # per-frame change below this on every joint (action and state) = still
MIN_PAUSE = 8     # still runs this long or longer (frames) inside an episode get compressed
KEEP_PAUSE = 2    # frames kept on each side of a compressed pause
KEEP_EDGE = 5     # frames kept before first / after last motion
MAX_CUT = 0.5     # never remove more than this fraction of an episode (falls back to edge trim)
WORKERS = 8


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def setc(t, name, arr):
    return t.set_column(t.schema.get_field_index(name), name, arr)


def masks(s, a):
    n = len(s)
    d = np.zeros(n)
    d[1:] = np.maximum(np.abs(np.diff(a, axis=0)).max(1), np.abs(np.diff(s, axis=0)).max(1))
    still = d < IDLE
    mv = np.flatnonzero(~still)
    edge = np.ones(n, bool)
    if not len(mv):
        return edge, edge, []
    edge[:max(0, mv[0] - KEEP_EDGE)] = False
    edge[mv[-1] + 1 + KEEP_EDGE:] = False
    full, cut = edge.copy(), []
    m = np.diff(np.r_[0, still.astype(int), 0])
    for x, y in zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)):
        if x > mv[0] and y <= mv[-1] and y - x >= MIN_PAUSE:
            full[x + KEEP_PAUSE:y - KEEP_PAUSE] = False
            cut.append([int(x), int(y)])
    return edge, full, cut


def pick_codec():
    for name in ("libx264", "h264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            pass
    raise SystemExit("no usable video encoder in PyAV")


def reencode(src, dst, frame_ids, fps, codec):
    with av.open(src) as ic:
        frames = [f.to_ndarray(format="rgb24") for f in ic.decode(video=0)]
    sel = [frames[min(k, len(frames) - 1)] for k in frame_ids]
    h, w = sel[0].shape[:2]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with av.open(dst, "w") as oc:
        st = oc.add_stream(codec, rate=fps)
        st.width, st.height = w, h
        st.pix_fmt = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"
        if codec == "libx264":
            st.options = {"crf": "20", "preset": "veryfast"}
        st.codec_context.gop_size = 2
        for img in sel:
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                oc.mux(pkt)
        for pkt in st.encode():
            oc.mux(pkt)
    return len(frames), st.pix_fmt


def work(job):
    d, out, info, cams, ep, idx, off, codec = job
    cs, fps = info["chunks_size"], info["fps"]
    rel = info["data_path"].format(episode_chunk=ep // cs, episode_index=ep)
    t = pq.read_table(f"{d}/{rel}")
    md = t.schema.metadata
    src_frames = np.rint(np.array(t["timestamp"].to_pylist()) * fps).astype(int)[idx]
    t = t.take(pa.array(idx))
    m = t.num_rows
    t = setc(t, "timestamp", pa.array(np.arange(m) / fps, t.schema.field("timestamp").type))
    t = setc(t, "frame_index", pa.array(range(m), t.schema.field("frame_index").type))
    t = setc(t, "index", pa.array(range(off, off + m), t.schema.field("index").type))
    os.makedirs(os.path.dirname(f"{out}/{rel}"), exist_ok=True)
    pq.write_table(t.replace_schema_metadata(md), f"{out}/{rel}")
    pix = {}
    for c in cams:
        vr = info["video_path"].format(episode_chunk=ep // cs, video_key=c, episode_index=ep)
        nsrc, pix[c] = reencode(f"{d}/{vr}", f"{out}/{vr}", src_frames, fps, codec)
    s = np.array(t["observation.state"].to_pylist(), np.float32)
    a = np.array(t["action"].to_pylist(), np.float32)
    st = {k: {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(),
              "std": v.std(0).tolist(), "count": [int(m)]} for k, v in (("observation.state", s), ("action", a))}
    return ep, m, st, pix


if __name__ == "__main__":
    codec = pick_codec()
    print("encoder:", codec)
    for src in SOURCES:
        d, out = f"{SRC_ROOT}/{src}", f"{OUT_ROOT}/{src}"
        info = json.load(open(f"{d}/meta/info.json"))
        cs = info["chunks_size"]
        cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        eps = jl(f"{d}/meta/episodes.jsonl")
        stats = {x["episode_index"]: x for x in jl(f"{d}/meta/episodes_stats.jsonl")}
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(f"{out}/meta")
        jobs, report, off = [], {}, 0
        for e in eps:
            i = e["episode_index"]
            rel = info["data_path"].format(episode_chunk=i // cs, episode_index=i)
            t = pq.read_table(f"{d}/{rel}", columns=["observation.state", "action"])
            s = np.array(t["observation.state"].to_pylist(), np.float32)
            a = np.array(t["action"].to_pylist(), np.float32)
            edge, full, cut = masks(s, a)
            mode = "full"
            if full.sum() < (1 - MAX_CUT) * len(s):
                full, mode = edge, "edge_only"
            idx = np.flatnonzero(full)
            report[i] = {"before": len(s), "after": int(len(idx)), "mode": mode, "pauses_cut": cut}
            jobs.append((d, out, info, cams, i, idx, off, codec))
            off += len(idx)
        results = {}
        with ProcessPoolExecutor(WORKERS) as ex:
            for ep, m, st, pix in ex.map(work, jobs):
                results[ep] = (m, st, pix)
                print(f"  {src[:24]} ep{ep:3d}: {report[ep]['before']} -> {m} frames "
                      f"({len(report[ep]['pauses_cut'])} pauses, {report[ep]['mode']})", flush=True)
        new_eps, new_stats = [], []
        for e in eps:
            i = e["episode_index"]
            m, st, pix = results[i]
            new_eps.append({**e, "length": m})
            s2 = json.loads(json.dumps(stats[i]))
            s2["stats"].update(st)
            new_stats.append(s2)
        info2 = json.loads(json.dumps(info))
        info2["total_frames"] = off
        for c in cams:
            vi = info2["features"][c].setdefault("info", {})
            vi["video.codec"] = "h264" if codec in ("libx264", "h264") else codec
            vi["video.pix_fmt"] = next(iter(results.values()))[2][c]
        json.dump(info2, open(f"{out}/meta/info.json", "w"), indent=2)
        open(f"{out}/meta/episodes.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_eps)
        open(f"{out}/meta/episodes_stats.jsonl", "w").writelines(json.dumps(x) + "\n" for x in new_stats)
        for f in ("tasks.jsonl", "conversion.json"):
            if os.path.exists(f"{d}/meta/{f}"):
                shutil.copy(f"{d}/meta/{f}", f"{out}/meta/{f}")
        json.dump({"params": {"IDLE": IDLE, "MIN_PAUSE": MIN_PAUSE, "KEEP_PAUSE": KEEP_PAUSE,
                              "KEEP_EDGE": KEEP_EDGE, "MAX_CUT": MAX_CUT}, "episodes": report},
                  open(f"{out}/meta/clean_report.json", "w"), indent=1)
        b = sum(r["before"] for r in report.values())
        print(f"== {src}: {b} -> {off} frames ({100 * (1 - off / b):.0f}% removed), "
              f"{sum(len(r['pauses_cut']) for r in report.values())} pauses compressed, "
              f"{sum(r['mode'] == 'edge_only' for r in report.values())} eps edge-only\n")
