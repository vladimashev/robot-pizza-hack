"""Evaluate failure checker v2 against the fixed roi.json boxes (v1) on the held-out rounds (round % 5 == 0, as in
h_train_change.py), with and without the frames shifted (np.roll) to simulate a misplaced pizza.
Pairs per episode, as in g_pair_dataset.py but all combinations: before = frames 0-2, after = last 8 frames;
target sector -> added, other sectors and before/before, after/after -> not added. GAP episodes skipped.
detected = pepperoni added and p > thr; false_added = no pepperoni added and p > thr (cheese episodes included).
Modes (located = where the pizza is found, radius always 28):
    fixed           roi.json boxes (v1)
    ep-raw          located on frame 0 of each episode, no bias correction, no dead zone (first version)
    round-raw       located once per round (frame 0 of the round's first episode, as live), no correction
    round           per round, offset taken from REF_LOCATED (bias corrected), no dead zone
    round-dz        per round, bias corrected, dead zone 4 px  (= FailureCheckerV2 defaults)
    ep-dz           per episode, bias corrected, dead zone 4 px
Usage: python t_eval_v2.py [thr ...]    -> tables + ~/v2/checker/v2_eval.csv"""
import json, os, sys
import av, numpy as np, pandas as pd

sys.path.insert(0, "/home/jovyan/v2/scripts")
from failure_checker_v2 import REF_LOCATED, FailureCheckerV2

H = "/home/jovyan"
ROOT, MAN = f"{H}/v2/lerobot", f"{H}/datasets"
TOPPING = {"cheese-r1r10-2026-10-24": "cheese", "pepperoni-r1r10-2026-10-24": "pepperoni",
           "pepperoni-on-cheese-r1r5-2026-10-24": "pepperoni"}
SHIFTS = [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5), (5, 5), (-5, -5), (8, 0), (0, -8)]
MODES = {"fixed": None, "ep-raw": ("ep", None, 0), "round-raw": ("round", None, 0), "round": ("round", REF_LOCATED, 0),
         "round-dz": ("round", REF_LOCATED, 4), "ep-dz": ("ep", REF_LOCATED, 4)}
THRS = [float(x) for x in sys.argv[1:]] or [0.5, 0.65]
BEFORE, AFTER_N = [0, 1, 2], 8
excl = json.load(open(f"{H}/v2/exclude.json")) if os.path.exists(f"{H}/v2/exclude.json") else {}

fc = FailureCheckerV2(device="cuda")
fc.thr = 0.0                                                # keep every probability, threshold below
episodes, round_f0 = [], {}
for src, topping in TOPPING.items():
    d = f"{ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    man = {e["episode_start_ns"]: e for e in json.load(open(f"{MAN}/{src}.manifest.json"))["episodes"]}
    conv = {c["episode_index"]: c for c in json.load(open(f"{d}/meta/conversion.json"))["episodes"]}
    for e in map(json.loads, open(f"{d}/meta/episodes.jsonl")):
        i = e["episode_index"]
        m = man.get(conv[i]["source_episode_start_ns"], {})
        rnd = m.get("round") or 0
        if rnd % 5:
            continue
        vp = f"{d}/" + info["video_path"].format(episode_chunk=i // info["chunks_size"],
                                                 video_key="observation.images.top", episode_index=i)
        with av.open(vp) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        round_f0.setdefault((src, rnd), fr[0])              # run start: first episode of the round
        if m.get("target") not in fc.fixed_boxes or i in excl.get(src, []):
            continue
        keep = BEFORE + list(range(len(fr) - AFTER_N, len(fr)))
        episodes.append(dict(src=src, ep=i, round=rnd, target=m["target"], topping=topping,
                             frames={k: fr[k] for k in keep}, after=keep[len(BEFORE):]))
print(f"{len(episodes)} held-out episodes in {len(round_f0)} rounds")


def scan_pairs(e, F, boxes):
    """[(sector, added, p)] for all pairs of one episode with the given boxes."""
    fc.checker.roi = boxes
    a = e["after"]
    pairs = [(b, x, True) for b in BEFORE for x in a]
    pairs += [(b, x, False) for b, x in [(0, 1), (1, 2), (0, 2), (a[0], a[-1]), (a[2], a[5]), (a[4], a[7])]]
    out = []
    for b, x, real in pairs:
        for s, v in fc.checker.scan(F[b], F[x]).items():
            if real or s == e["target"]:                    # static pairs: target sector only, as in training
                out.append((s, real and s == e["target"] and e["topping"] == "pepperoni", v["pepperoni"]))
    return out


rows, starts = [], []
for dx, dy in SHIFTS:
    sh = f"{dx:+d},{dy:+d}"
    roll = lambda f: np.roll(f, (dy, dx), axis=(0, 1))
    for e in episodes:
        F = {k: roll(f) for k, f in e["frames"].items()}
        cache = {}
        for mode, how in MODES.items():
            if how is None:
                st = fc.use_fixed()
            else:
                where, ref, dz = how
                fc.ref_located, fc.dead_zone_px = list(ref or fc.ref_center), dz
                st = fc.start(F[0] if where == "ep" else roll(round_f0[e["src"], e["round"]]))
                starts.append(dict(shift=sh, mode=mode, src=e["src"], ep=e["ep"], source=st["boxes_source"],
                                   fallback=st["fallback"], offset=st["offset_px"], ms=st["ms"]))
            key = json.dumps(st["boxes"], sort_keys=True)
            if key not in cache:
                cache[key] = scan_pairs(e, F, st["boxes"])
            rows += [dict(shift=sh, mode=mode, src=e["src"], ep=e["ep"], sector=s, added=ad, p=p)
                     for s, ad, p in cache[key]]
    print(f"shift {sh} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{H}/v2/checker/v2_eval.csv", index=False)
out = []
for (shift, mode), g in df.groupby(["shift", "mode"], sort=False):
    pos, neg = g[g.added], g[~g.added]
    row = dict(shift=shift, mode=mode)
    for t in THRS:
        row[f"detected@{t}"] = f"{(pos.p > t).sum()}/{len(pos)}"
        row[f"false_added@{t}"] = f"{(neg.p > t).sum()}/{len(neg)}"
    out.append(row)
print("\n=== pepperoni, held-out rounds ===")
print(pd.DataFrame(out).to_string(index=False))
S = pd.DataFrame(starts)
print("\n=== v2 start(): boxes used per mode and shift (episodes) ===")
print(S.groupby(["shift", "mode"], sort=False).agg(
    dynamic=("source", lambda s: int((s == "dynamic").sum())), fallback=("fallback", "sum"),
    offset_median=("offset", "median"), ms_median=("ms", "median")).round(1).to_string())
