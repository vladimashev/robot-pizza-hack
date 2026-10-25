"""Find each target's region in the top camera from first-vs-last frame differences.
Reads ~/v2/lerobot (read-only). Writes ~/v2/checker/roi.json + heatmaps for visual check."""
import collections, json, os
import numpy as np, av
from PIL import Image, ImageDraw, ImageFilter

H = os.path.expanduser("~")
ROOT, MAN, OUT = f"{H}/v2/lerobot", f"{H}/datasets", f"{H}/v2/checker"
SOURCES = ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24", "pepperoni-on-cheese-r1r5-2026-10-24"]
TOP = "observation.images.top"
SKIP = {"GAP", None}
os.makedirs(OUT, exist_ok=True)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def target_map(src, d):
    man = {e["episode_start_ns"]: e for e in json.load(open(f"{MAN}/{src}.manifest.json"))["episodes"]}
    conv = json.load(open(f"{d}/meta/conversion.json"))["episodes"]
    return {c["episode_index"]: man.get(c["source_episode_start_ns"], {}).get("target") for c in conv}


acc, cnt, sample = {}, collections.Counter(), None
for src in SOURCES:
    d = f"{ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    tmap = target_map(src, d)
    for e in jl(f"{d}/meta/episodes.jsonl"):
        i = e["episode_index"]
        tg = tmap.get(i)
        if tg in SKIP:
            continue
        vp = f"{d}/" + info["video_path"].format(episode_chunk=i // info["chunks_size"], video_key=TOP, episode_index=i)
        with av.open(vp) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        diff = np.abs(fr[-1].astype(np.float32) - fr[0].astype(np.float32)).mean(2)
        acc[tg] = diff if tg not in acc else acc[tg] + diff
        cnt[tg] += 1
        if sample is None:
            sample = fr[0]

roi = {}
h, w = sample.shape[:2]
overview = Image.fromarray(sample)
dr = ImageDraw.Draw(overview)
for tg in sorted(acc):
    m = acc[tg] / cnt[tg]
    m8 = Image.fromarray((255 * m / m.max()).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    m = np.asarray(m8, np.float32)
    ys, xs = np.nonzero(m >= np.percentile(m, 97))
    x0, x1 = np.percentile(xs, [5, 95])
    y0, y1 = np.percentile(ys, [5, 95])
    pad = 0.15 * max(x1 - x0, y1 - y0)
    box = [int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad))]
    roi[tg] = box
    heat = m8.convert("RGB")
    ImageDraw.Draw(heat).rectangle(box, outline=(255, 0, 0), width=3)
    heat.save(f"{OUT}/heat_{tg}.png")
    dr.rectangle(box, outline=(255, 0, 0), width=3)
    dr.text((box[0] + 4, box[1] + 4), tg, fill=(255, 255, 0))
    print(f"{tg:7} n={cnt[tg]:3d}  box={box}")
overview.save(f"{OUT}/roi_overview.jpg")
json.dump({"image_size": [w, h], "roi": roi}, open(f"{OUT}/roi.json", "w"), indent=1)
print(f"saved {OUT}/roi.json, roi_overview.jpg, heat_*.png")
