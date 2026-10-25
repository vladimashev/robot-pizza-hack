"""Episode quality analysis for pizza-topping datasets. Read-only on inputs.
Inputs : ~/v2/lerobot/<src>  (converted LeRobot v2.1), ~/datasets/<src>.manifest.json
Outputs: ~/v2/analysis/  episodes.csv, flagged.csv, <src>_gripper.png, <src>_grasp_wrist.jpg, <src>_end_top.jpg
Usage  : python analyze.py [src ...]
Assumes gripper = state/action dim 7 and LOWER value = MORE closed (checked at runtime)."""
import json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from PIL import Image, ImageDraw
import av

H = os.path.expanduser("~")
ROOT = os.environ.get("ROOT", f"{H}/v2/lerobot"); MAN = f"{H}/datasets"; OUT = os.environ.get("OUT", f"{H}/v2/analysis")
assert not os.path.realpath(OUT).startswith(os.path.realpath(MAN))
SOURCES = sys.argv[1:] or ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24",
                           "pepperoni-on-cheese-r1r5-2026-10-24"]
G = 7                          # gripper dim
CLOSE_TH, OPEN_TH = 0.60, 0.70 # hysteresis on commanded gripper (action)
IDLE = 0.005                   # per-frame joint motion below this = not moving (norm units)
PAUSE_S = 1.0                  # mid-episode stillness longer than this counts as a pause
EMPTY_GAP = 0.02               # state - action while closed below this -> possibly empty grasp
WRIST, TOP = "observation.images.front", "observation.images.top"
os.makedirs(OUT, exist_ok=True)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def runs(mask):
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def closed_segments(g):
    segs, closed, start = [], False, 0
    for i, v in enumerate(g):
        if not closed and v < CLOSE_TH:
            closed, start = True, i
        elif closed and v > OPEN_TH:
            closed = False
            segs.append((start, i))
    if closed:
        segs.append((start, len(g)))
    return segs


def frame_at(path, ts, width=220):
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            c.seek(int(max(ts - 0.5, 0) / s.time_base), stream=s, backward=True)
            img = None
            for f in c.decode(s):
                img = f
                if float(f.pts * s.time_base) >= ts - 1e-3:
                    break
            im = img.to_image()
    except Exception:
        im = Image.new("RGB", (320, 240), "gray")
    return im.resize((width, int(im.height * width / im.width)))


def sheet(items, path, cols=8):
    if not items:
        return
    w, h = items[0][0].size
    lab = 28
    canvas = Image.new("RGB", (cols * w, ((len(items) + cols - 1) // cols) * (h + lab)), "white")
    dr = ImageDraw.Draw(canvas)
    for k, (im, text, bad) in enumerate(items):
        x, y = (k % cols) * w, (k // cols) * (h + lab)
        canvas.paste(im, (x, y + lab))
        dr.text((x + 3, y + 2), text, fill=(200, 0, 0) if bad else (0, 0, 0))
    canvas.save(path, quality=85)


rows, traces = [], {}
for src in SOURCES:
    d = f"{ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    fps, cs = info["fps"], info["chunks_size"]
    man = {e["episode_start_ns"]: e for e in json.load(open(f"{MAN}/{src}.manifest.json"))["episodes"]}
    conv_p = f"{d}/meta/conversion.json"
    conv = {c["episode_index"]: c for c in json.load(open(conv_p))["episodes"]} if os.path.exists(conv_p) else {}
    traces[src] = []
    for e in jl(f"{d}/meta/episodes.jsonl"):
        i = e["episode_index"]
        p = f"{d}/" + info["data_path"].format(episode_chunk=i // cs, episode_index=i)
        t = pq.read_table(p, columns=["observation.state", "action", "timestamp"])
        s = np.array(t["observation.state"].to_pylist(), np.float32)
        a = np.array(t["action"].to_pylist(), np.float32)
        ts = np.array(t["timestamp"].to_pylist())
        n = len(s)
        gs, ga = s[:, G], a[:, G]
        segs = [(x, y) for x, y in closed_segments(ga) if y - x >= 3]
        main = max(segs, key=lambda z: z[1] - z[0]) if segs else None
        gap = float(np.median(gs[main[0]:main[1]] - ga[main[0]:main[1]])) if main else np.nan
        mv = np.r_[0, np.abs(np.diff(a[:, :G], axis=0)).max(1)] > IDLE
        mi = np.flatnonzero(mv)
        pauses = [(y - x) / fps for x, y in runs(~mv[mi[0]:mi[-1]])] if len(mi) else []
        m = man.get(conv.get(i, {}).get("source_episode_start_ns"), {})
        quad = m.get("target") or "?"
        rows.append(dict(
            src=src, ep=i, name=m.get("id", ""), round=m.get("round", ""), target=quad,
            seconds=n / fps, n_grasps=len(segs),
            first_grasp_s=main[0] / fps if main else np.nan,
            hold_s=(main[1] - main[0]) / fps if main else np.nan,
            grip_gap=gap, grip_min_cmd=float(ga.min()), start_grip=float(gs[0]),
            idle_start_s=mi[0] / fps if len(mi) else n / fps,
            idle_end_s=(n - 1 - mi[-1]) / fps if len(mi) else 0.0,
            n_pauses=sum(x >= PAUSE_S for x in pauses), max_pause_s=max(pauses, default=0.0),
            path=float(np.abs(np.diff(a[:, :G], axis=0)).sum()),
            _ts=ts, _main=main, _info=info, _d=d))
        traces[src].append((np.linspace(0, 1, n), gs, quad))

df = pd.DataFrame(rows)
if (df.target == "?").all():
    print("WARNING: could not match episodes to manifest (conversion.json missing?) -> target unknown")
if (df.start_grip < df.grip_min_cmd).mean() > 0.5:
    print("WARNING: gripper seems inverted (higher = closed). Grasp stats unreliable.")

for c in ["seconds", "path"]:
    df[f"z_{c}"] = df.groupby(["src", "target"])[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
typical = df.groupby("src").n_grasps.transform(lambda v: v.mode().iloc[0])
flags = {
    "no_grasp": df.n_grasps == 0,
    "extra_grasps": df.n_grasps > typical,
    "maybe_empty": df.grip_gap < EMPTY_GAP,
    "closed_start": df.start_grip < OPEN_TH,
    "long_pause": df.max_pause_s > 2 * PAUSE_S,
    "len_outlier": df.z_seconds.abs() > 2,
    "path_outlier": df.z_path > 2,
}
df['issues'] = [",".join(k for k, v in flags.items() if v.iloc[r]) for r in range(len(df))]

pub = df[[c for c in df.columns if not c.startswith("_")]]
pub.to_csv(f"{OUT}/episodes.csv", index=False)
pub[pub['issues'] != ""].to_csv(f"{OUT}/flagged.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== per dataset ===")
print(df.groupby("src").agg(
    eps=("ep", "count"), sec_med=("seconds", "median"),
    grasp_counts=("n_grasps", lambda v: dict(v.value_counts().sort_index())),
    gap_med=("grip_gap", "median"),
    pct_maybe_empty=("grip_gap", lambda v: round(100 * (v < EMPTY_GAP).mean())),
    pct_closed_start=("start_grip", lambda v: round(100 * (v < OPEN_TH).mean())),
    idle_start_med=("idle_start_s", "median"), idle_end_med=("idle_end_s", "median"),
    pauses_mean=("n_pauses", "mean")).round(2).to_string())
print("\n=== per dataset x target ===")
print(df.groupby(["src", "target"]).agg(
    eps=("ep", "count"), sec=("seconds", "median"),
    extra=("issues", lambda v: sum("extra_grasps" in f for f in v)),
    empty=("issues", lambda v: sum("maybe_empty" in f for f in v)),
    flagged=("issues", lambda v: sum(f != "" for f in v))).to_string())
print(f"\nflagged {int((df['issues'] != '').sum())}/{len(df)} episodes -> {OUT}/flagged.csv")
for _, r in df[df['issues'] != ""].iterrows():
    print(f"  {r.src[:22]:22} ep{r.ep:3d} {str(r['name']):20} {r.target:7} {r['issues']}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"TL": "C0", "TR": "C1", "BR": "C2", "BL": "C3", "CENTER": "C4", "GAP": "C5", "?": "k"}
    for src, tr in traces.items():
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        for x, g, q in tr:
            ax[0].plot(x, g, color=colors.get(q, "k"), alpha=0.4, lw=1)
        ax[0].axhline(CLOSE_TH, ls="--", c="gray"); ax[0].axhline(OPEN_TH, ls=":", c="gray")
        ax[0].set(title=f"{src}: gripper state (color = target)", xlabel="episode progress", ylabel="gripper norm")
        ax[1].hist(df[df.src == src].grip_gap.dropna(), bins=30); ax[1].axvline(EMPTY_GAP, c="r", ls="--")
        ax[1].set(title="grip gap while closed (state - command)", xlabel="gap")
        fig.tight_layout(); fig.savefig(f"{OUT}/{src}_gripper.png", dpi=90); plt.close(fig)
except ImportError:
    print("matplotlib missing: pip install matplotlib  (plots skipped)")

for src in SOURCES:
    grasp, end = [], []
    for _, r in df[df.src == src].iterrows():
        vp = lambda key: f"{r._d}/" + r._info["video_path"].format(
            episode_chunk=r.ep // r._info["chunks_size"], video_key=key, episode_index=r.ep)
        label = f"{r['name'] or r.ep} {r.target} {r['issues'][:26]}"
        k = min(r._main[0] + 5, r._main[1] - 1) if r._main else len(r._ts) // 2
        grasp.append((frame_at(vp(WRIST), float(r._ts[k])), label, bool(r['issues'])))
        end.append((frame_at(vp(TOP), float(r._ts[-1])), label, bool(r['issues'])))
    sheet(grasp, f"{OUT}/{src}_grasp_wrist.jpg")
    sheet(end, f"{OUT}/{src}_end_top.jpg")
print(f"\nimages: {OUT}/*_gripper.png, *_grasp_wrist.jpg, *_end_top.jpg")
