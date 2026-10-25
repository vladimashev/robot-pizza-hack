"""Build crop dataset for the topping checker: target-sector crops before (first frames) and after
(last frames) each episode. Reads ~/v2/lerobot (read-only). Writes ~/v2/checker/crops + labels.csv."""
import json, os, shutil
import av, pandas as pd
from PIL import Image

H = os.path.expanduser("~")
ROOT, MAN, OUT = f"{H}/v2/lerobot", f"{H}/datasets", f"{H}/v2/checker"
TOP = "observation.images.top"
LABELS = {  # src -> ((cheese, pepperoni) before, (cheese, pepperoni) after)
    "cheese-r1r10-2026-10-24": ((0, 0), (1, 0)),
    "pepperoni-r1r10-2026-10-24": ((0, 0), (0, 1)),
    "pepperoni-on-cheese-r1r5-2026-10-24": ((1, 0), (1, 1)),
}
BEFORE = [0, 1, 2]
AFTER_N = 8
SIZE = 160
roi = json.load(open(f"{OUT}/roi.json"))["roi"]
excl = json.load(open(f"{H}/v2/exclude.json")) if os.path.exists(f"{H}/v2/exclude.json") else {}
shutil.rmtree(f"{OUT}/crops", ignore_errors=True)
os.makedirs(f"{OUT}/crops")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rows = []
for src, (lab_b, lab_a) in LABELS.items():
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
        n = len(fr)
        for kind, idxs, lab in (("before", BEFORE, lab_b), ("after", range(n - AFTER_N, n), lab_a)):
            for k in idxs:
                p = f"{OUT}/crops/{src[:14]}_{i:03d}_{kind}_{k:04d}.png"
                Image.fromarray(fr[k]).crop(roi[tg]).resize((SIZE, SIZE), Image.BICUBIC).save(p)
                rows.append(dict(path=p, src=src, ep=i, round=m.get("round", 0), target=tg, kind=kind,
                                 has_cheese=lab[0], has_pepperoni=lab[1]))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/labels.csv", index=False)
print(df.groupby(["src", "kind"]).size().to_string())
print(df.groupby("target").size().to_string())
print(f"\n{len(df)} crops from {df.groupby(['src', 'ep']).ngroups} episodes -> {OUT}/labels.csv")
