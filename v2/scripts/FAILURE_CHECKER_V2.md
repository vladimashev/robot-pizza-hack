# Failure checker v2: change checker with a located pizza

v1 (`change_checker.py`) decides after a pepperoni step whether a slice was **added** to the target sector,
using **fixed** sector boxes (`roi.json`, pizza circle `118,131 r28`). v2 uses the same model and the same
decision, but at the start of a run it finds the pizza in the top-camera frame and moves the five sector
boxes with it. The radius stays 28 px (the scale `change.pt` was trained at).

## Files

| What | Path |
|---|---|
| v2 runtime | `~/v2/scripts/failure_checker_v2.py` (`FailureCheckerV2`) |
| Evaluation | `~/v2/scripts/t_eval_v2.py` -> `~/v2/checker/v2_eval.csv`, log `v2_eval.log` |
| Model, sector reference | `~/v2/checker/change.pt`, `~/v2/checker/roi.json` (must be r28; backup `roi_r28.json`) |
| Locator | `~/v2/scripts/pizza_locator.py` (method `red`) |
| Locator reference frame | frame 0 of `cheese-r1r10-2026-10-24` ep 0 (also saved as `~/v2/checker/ref_top.png`) |
| Robot runtime (laptop) | norma-core `software/ai/smolvla_py`: `smolvla/failure_checker_v2.py` + `smolvla/pizza_locator.py` (copies), `pizza/robot.py`, `scripts/run_challenge.py --failure-checker v2`, web UI `scripts/web_ui.py` |

## How it works

1. `start(top)` once per run, arm at home, clean or cheese-only pizza (no pepperoni yet, nothing over the pizza).
   `PizzaLocator(method="red")` finds the sauce region (Lab a\*, Otsu) and returns its center; the radius it
   estimates is ignored.
2. The located center is compared with `REF_LOCATED = (118.0, 133.8)`: where the locator puts the pizza at the
   start of the 20 training rounds (median). That is the pizza the fixed boxes were trained on, so
   `offset = located - REF_LOCATED`. Without this correction the locator is ~3 px off in y on the training
   data itself (it is calibrated on one frame, round starts jitter 2-3 px).
3. `offset <= dead_zone_px` (4 px): keep the fixed boxes (noise, and the model was trained on exactly them).
   `offset > max_offset_px` (15 px) or pizza not found: fixed boxes, `fallback: true`.
   Otherwise the boxes are `PizzaLocator.sector_boxes(118 + dx, 131 + dy, 28)`: same geometry as training.
4. `check_step(before, after, target, "pepperoni")` scans the five sectors with `ChangeChecker` and decides
   `p(target) > thr` (0.65). Cheese and GAP return `supported: false` (cheese is not reliable).

```python
from failure_checker_v2 import FailureCheckerV2
fc = FailureCheckerV2()                       # radius=28, max_offset_px=15, dead_zone_px=4, thr=0.65
status = fc.start(top)                        # dict, see below
res = fc.check_step(before, after, "TL")      # dict, see below
overlay = fc.render(after, res)               # RGB, 3x upscaled: circle, boxes, target green/red, p per sector
fc.use_fixed()                                # force the v1 boxes
```

### Data contract (JSON)

```json
// start()
{"version": "v2", "method": "red", "radius": 28, "boxes_source": "dynamic", "found": true, "fallback": false,
 "reason": "", "center": [137.5, 147.3], "offset_px": 25.4, "shift_px": [19.5, 13.5], "ms": 0.6,
 "boxes": {"TL": [x0, y0, x1, y1], "TR": [...], "BR": [...], "BL": [...], "CENTER": [...]}}
// check_step()
{"version": "v2", "supported": true, "target": "TL", "topping": "pepperoni", "p": 0.97, "ok": true,
 "threshold": 0.65, "per_sector": {"TL": 0.97, "TR": 0.01, "BR": 0.0, "BL": 0.02, "CENTER": 0.01},
 "elsewhere": {}, "boxes_source": "dynamic", "ms": 12.0}
```
`center` is in roi.json terms (the sector circle the boxes are built around). `boxes_source` is `fixed`
inside the dead zone and on fallback; `reason` says why.

## Validation

Held-out rounds (round % 5 == 0: cheese R5/R10, pepperoni R5/R10, pepperoni-on-cheese R5; 29 episodes, GAP
skipped), all before x after pairs: 360 pepperoni-added pairs, 3294 not-added pairs (other sectors, static
pairs, cheese episodes). `python t_eval_v2.py` (~45 min on gpu2). Threshold 0.65, `detected / false_added`:

| shift (px) | fixed (v1) | v2 as first specified (per episode, no correction) | **v2 (per round, corrected, dead zone 4)** |
|---|---|---|---|
| 0, 0   | 357 / 4   | 326 / 10 | **357 / 4** |
| +5, 0  | 329 / 68  | 319 / 30 | **346 / 42** |
| -5, 0  | 325 / 42  | 326 / 22 | **329 / 0** |
| 0, +5  | 336 / 17  | 326 / 10 | **353 / 0** |
| 0, -5  | 333 / 55  | 326 / 34 | **333 / 44** |
| +5, +5 | 310 / 48  | 322 / 6  | **349 / 10** |
| -5, -5 | 336 / 93  | 326 / 34 | **329 / 47** |
| +8, 0  | 288 / 73  | 302 / 30 | **329 / 24** |
| 0, -8  | 330 / 108 | 326 / 34 | **329 / 55** |

(shift = `np.roll` of all frames incl. the localization frame; at thr 0.5 the picture is the same, see `v2_eval.log`.)

- Without a shift, v2 is identical to v1: every round start lands inside the 4 px dead zone (median offset 1.3 px).
- Locating per episode with no correction (the first version) loses 31 detections: the red center of a cheese-covered
  pizza jumps 5-7 px in single episodes (pepperoni-on-cheese R5 BL: 24/24 -> 0/24), and it carries a ~3 px y bias.
  That is why v2 locates once per round/run, measures the offset from `REF_LOCATED` and has the dead zone.
- Shifted: v2 has fewer false "added" than v1 in every case and equal or more detections in 7 of 8; -5,-5 loses
  7 detections (but halves the false adds). The remaining errors are mostly the 5 pepperoni-on-cheese R5 episodes,
  where the locator does not find the cheese-covered pizza at the round start -> fallback to the fixed boxes.
- Speed: `start()` 0.3-0.6 ms, `check_step()` ~10 ms on GPU, ~100 ms on CPU (5 sectors).
- Fallback (`start()` on bad frames): black, white, noise, pizza blacked out, 112x112 crop, zoomed crop, an empty
  float array: all `fallback: true` with the fixed boxes, no exception.
- Live (laptop, 2026-10-25, read-only dry run): (1) a hand over the pizza: located 15.5 px off -> fallback;
  (2) clear scene, pizza in its usual spot: 1.1 px -> fixed boxes; (3) the pizza had been moved: located
  25.4 px off (dx +19.5, dy +13.5), circle and boxes visually on the pizza; fallback at the default 15 px,
  dynamic boxes with `--checker-max-offset 30` (overlay: smolvla_py `logs/checker_v2_live_start.png`).

## Robot / UI

On the laptop (`software/ai/smolvla_py`):

```bash
uv run python scripts/run_challenge.py ch2 --failure-checker v2
uv run python scripts/run_challenge.py ch2 --failure-checker v2 --dry-run   # locate on the live image, arm not moved
uv run python scripts/run_challenge.py ch2 --failure-checker v2 --checker-max-offset 30
```
`PizzaRobot.connect()` glides home, takes the at-home scene frame and calls `start()`; the existing retry loop
(`pizza/recipe.py run_steps`, `--retries 2`) uses the v2 verdict. `--checker-log DIR` writes `events.jsonl`
(`start`, `check`, `decision` with attempt / retry) and overlay PNGs; the web UI sets it for every run and shows
a "Failure checker" panel: start overlay + found / fallback / offset, then one card per checked step (target,
p, OK / MISSING, elsewhere, attempt n/max, retry / gave up). The UI has a "Checker version" select (v1 / v2).

The gpu2 runner `~/scripts/run_lerobot_policy.py` has no checker or retry loop, so it was not changed.

## Limitations

- Localization must run with nothing over the pizza: an arm or a hand is red in Lab a\* and pulls the center
  (live, a hand over the pizza gave 15.5 px -> fallback).
- Run it once at the start: pepperoni and heavy cheese shift the red-region center (cheese round starts already
  jitter 2-3 px, a few episodes 5-7 px).
- Offsets up to 4 px are not corrected (dead zone). Beyond 15 px it falls back to v1 unless the limit is raised;
  boxes moved by 20+ px were not validated offline (the shift test goes to 8 px), only checked visually live.
- The change model was trained at r28 sector scale and this camera pose; a pizza that looks much bigger/smaller
  (camera moved up/down) is out of distribution and needs retraining with scale augmentation.
- Pepperoni only; GAP has no box.
