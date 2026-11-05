"""Build the before/after pair dataset for the change checker (h_train_change.py reads pairs.csv).
Reads ~/v2/lerobot (read-only) + ~/datasets/<src>.manifest.json. Writes ~/v2/checker/pair_crops + pairs.csv.
Per episode (top camera, fixed roi.json boxes, crops resized to SIZE like change.pt "size"):
    before = frames 0-2, after = last 8 frames
    real pairs (each before x each after), every sector: target sector -> added topping of the source, others -> 0
    static pairs (0,1) (1,2) (0,2) and three after/after pairs, target sector only -> 0
GAP episodes and exclude.json episodes are skipped. Validation split (round % 5 == 0) is done in h_train_change.py.
Rebuilt from the pair definition in t_eval_v2.py after the original script was lost."""
import json, os, shutil
import av, pandas as pd
from PIL import Image

H = os.path.expanduser("~")
ROOT, MAN, OUT = f"{H}/v2/lerobot", f"{H}/datasets", f"{H}/v2/checker"
TOP = "observation.images.top"
ADDED = {  # src -> (added_cheese, added_pepperoni) in the target sector
    "cheese-r1r10-2026-10-24": (1, 0),
    "pepperoni-r1r10-2026-10-24": (0, 1),
    "pepperoni-on-cheese-r1r5-2026-10-24": (0, 1),
}
BEFORE, AFTER_N, SIZE = [0, 1, 2], 8, 160
roi = json.load(open(f"{OUT}/roi.json"))["roi"]
excl = json.load(open(f"{H}/v2/exclude.json")) if os.path.exists(f"{H}/v2/exclude.json") else {}
CROPS = f"{OUT}/pair_crops"
shutil.rmtree(CROPS, ignore_errors=True)
os.makedirs(CROPS)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rows = []
for src, (add_c, add_p) in ADDED.items():
    d = f"{ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    man = {e["episode_start_ns"]: e for e in json.load(open(f"{MAN}/{src}.manifest.json"))["episodes"]}
    conv = {c["episode_index"]: c for c in json.load(open(f"{d}/meta/conversion.json"))["episodes"]}
    for e in jl(f"{d}/meta/episodes.jsonl"):
        i = e["episode_index"]
        m = man.get(conv[i]["source_episode_start_ns"], {})
        tg = m.get("target")
        if tg not in roi or i in excl.get(src, []):
            continue
        vp = f"{d}/" + info["video_path"].format(episode_chunk=i // info["chunks_size"], video_key=TOP, episode_index=i)
        with av.open(vp) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        after = list(range(len(fr) - AFTER_N, len(fr)))
        path = {}                                          # (frame, sector) -> crop png, written once

        def crop(k, s):
            if (k, s) not in path:
                p = f"{CROPS}/{src[:14]}_{i:03d}_{s}_{k:04d}.png"
                Image.fromarray(fr[k]).crop(roi[s]).resize((SIZE, SIZE), Image.BICUBIC).save(p)
                path[k, s] = p
            return path[k, s]

        base = dict(src=src, ep=i, round=m.get("round", 0), target=tg)
        for b in BEFORE:
            for a in after:
                for s in roi:
                    hit = s == tg
                    rows.append(dict(base, before=crop(b, s), after=crop(a, s), sector=s,
                                     kind="target" if hit else "other_sector",
                                     added_cheese=add_c if hit else 0, added_pepperoni=add_p if hit else 0))
        for b, a in [(0, 1), (1, 2), (0, 2), (after[0], after[-1]), (after[2], after[5]), (after[4], after[7])]:
            rows.append(dict(base, before=crop(b, tg), after=crop(a, tg), sector=tg, kind="static",
                             added_cheese=0, added_pepperoni=0))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/pairs.csv", index=False)
print(df.groupby(["src", "kind"]).size().to_string())
print(df[["added_cheese", "added_pepperoni"]].sum().to_string())
print(f"\n{len(df)} pairs from {df.groupby(['src', 'ep']).ngroups} episodes -> {OUT}/pairs.csv")
