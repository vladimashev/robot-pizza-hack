"""Sector boxes from a manually given pizza circle; target->quadrant mapping learned from
first-vs-last frame differences INSIDE the pizza only. Reads ~/v2/lerobot (read-only).
Usage: python d_roi_manual.py CX CY R   -> ~/v2/checker/roi.json, roi_overview.jpg"""
import collections, json, os, sys
import numpy as np, av
from PIL import Image, ImageDraw

H = os.path.expanduser("~")
ROOT, MAN, OUT = f"{H}/v2/lerobot", f"{H}/datasets", f"{H}/v2/checker"
SOURCES = ["cheese-r1r10-2026-10-24", "pepperoni-r1r10-2026-10-24", "pepperoni-on-cheese-r1r5-2026-10-24"]
TOP = "observation.images.top"
QUADS = ["TL", "TR", "BR", "BL"]
cx, cy, r = (int(v) for v in sys.argv[1:4])
pad = max(2, int(0.08 * r))
os.makedirs(OUT, exist_ok=True)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


geo = {"img_UL": (cx - r, cy - r, cx, cy), "img_UR": (cx, cy - r, cx + r, cy),
       "img_LR": (cx, cy, cx + r, cy + r), "img_LL": (cx - r, cy, cx, cy + r)}
acc, cnt, sample = {}, collections.Counter(), None
for src in SOURCES:
    d = f"{ROOT}/{src}"
    info = json.load(open(f"{d}/meta/info.json"))
    man = {e["episode_start_ns"]: e for e in json.load(open(f"{MAN}/{src}.manifest.json"))["episodes"]}
    conv = {c["episode_index"]: c for c in json.load(open(f"{d}/meta/conversion.json"))["episodes"]}
    for e in jl(f"{d}/meta/episodes.jsonl"):
        i = e["episode_index"]
        tg = man.get(conv[i]["source_episode_start_ns"], {}).get("target")
        if tg not in QUADS + ["CENTER"]:
            continue
        vp = f"{d}/" + info["video_path"].format(episode_chunk=i // info["chunks_size"], video_key=TOP, episode_index=i)
        with av.open(vp) as c:
            fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        diff = np.abs(fr[-1].astype(np.float32) - fr[0].astype(np.float32)).mean(2)
        acc[tg] = diff if tg not in acc else acc[tg] + diff
        cnt[tg] += 1
        sample = fr[-1] if sample is None else sample

h, w = sample.shape[:2]
yy, xx = np.mgrid[0:h, 0:w]
dist = np.hypot(xx - cx, yy - cy)
ring = (dist <= r) & (dist >= 0.3 * r)
qmask = {k: ring & (xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3]) for k, b in geo.items()}

print("mean change inside each image quadrant, per target (each row should have one clear max):")
print(f"{'target':8}" + "".join(f"{k:>9}" for k in geo))
mapping = {}
for tg in QUADS:
    if tg not in acc:
        continue
    m = acc[tg] / cnt[tg]
    sc = {k: float(m[q].mean()) for k, q in qmask.items()}
    mapping[tg] = max(sc, key=sc.get)
    print(f"{tg:8}" + "".join(f"{sc[k]:9.2f}" for k in geo) + f"   -> {mapping[tg]}")
if len(set(mapping.values())) != len(mapping):
    print("\nWARNING: two targets map to the same image quadrant -> check CX/CY/R or the data")

clip = lambda b: [int(max(0, b[0] - pad)), int(max(0, b[1] - pad)), int(min(w, b[2] + pad)), int(min(h, b[3] + pad))]
roi = {tg: clip(geo[q]) for tg, q in mapping.items()}
c = 0.45 * r
roi["CENTER"] = clip((cx - c, cy - c, cx + c, cy + c))
json.dump({"image_size": [w, h], "pizza": [cx, cy, r], "mapping": mapping, "roi": roi},
          open(f"{OUT}/roi.json", "w"), indent=1)

m = int(1.5 * r)
z0, z1 = (max(0, cx - m), max(0, cy - m)), (min(w, cx + m), min(h, cy + m))
zoom = Image.fromarray(sample).crop((*z0, *z1))
s = max(1, 480 // zoom.width)
zoom = zoom.resize((zoom.width * s, zoom.height * s), Image.NEAREST)
dr = ImageDraw.Draw(zoom)
colors = {"TL": (255, 0, 0), "TR": (0, 255, 0), "BR": (0, 128, 255), "BL": (255, 255, 0), "CENTER": (255, 0, 255)}
for tg, b in roi.items():
    bb = [(b[0] - z0[0]) * s, (b[1] - z0[1]) * s, (b[2] - z0[0]) * s, (b[3] - z0[1]) * s]
    dr.rectangle(bb, outline=colors[tg], width=2)
    dr.text((bb[0] + 4, bb[1] + 4), tg, fill=colors[tg])
zoom.save(f"{OUT}/roi_overview.jpg")
print(f"\nroi: {roi}\nsaved {OUT}/roi.json and roi_overview.jpg (zoomed)")
