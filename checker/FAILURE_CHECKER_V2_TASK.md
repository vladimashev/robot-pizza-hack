# Task: Failure checker v2 (dynamic pizza localization) + UI integration

## Goal

The pepperoni failure checker (v1) decides after each policy step whether pepperoni was **added** to the
target pizza sector, using **fixed** sector boxes from `roi.json`. Build **v2**: at the start of every run
the pizza is located **dynamically** in the top-camera frame, the sector boxes are built around the
located center, and the same trained change model is used with these boxes. Then expose v2 in the UI.

Decisions already made by the user:
- localization method: **`red`** (sauce region by redness, `PizzaLocator._red` in `pizza_locator.py`);
- sector circle **radius is fixed to the default 28 px** (user checked visually; do not use the radius
  estimated by the locator, only its center);
- localization runs **once at the start of a run**, with the arm at home and a clean or cheese-only pizza;
  it is not recomputed during steps.

Keep v1 working and untouched.

## Existing pieces (DevLab, user `jovyan`)

| What | Path |
|---|---|
| Change model (trained with r=28 sector crops) | `/home/jovyan/v2/checker/change.pt` |
| Reference sectors: pizza circle `[118, 131, 28]`, quadrant mapping, boxes | `/home/jovyan/v2/checker/roi.json` |
| v1 runtime checker (`ChangeChecker`, attribute `roi` = dict of sector boxes) | `/home/jovyan/v2/scripts/change_checker.py` (+ `h_train_change.py`, same dir) |
| Pizza locator (`PizzaLocator`, methods `hough`, `red`, `sam`; `sector_boxes()`) | `/home/jovyan/v2/scripts/pizza_locator.py` |
| v1 documentation | `/home/jovyan/v2/scripts/CHANGE_CHECKER.md` |
| Locator comparison / evaluation scripts | `/home/jovyan/v2/scripts/q_compare_locators.py`, `r_eval_tracking.py`, `s_radius_grid.py` |
| Robot runner (has `--check-model` retry loop and `--box-guard`) | `/home/jovyan/scripts/run_lerobot_policy.py` |
| Converted datasets (top camera = `observation.images.top`) | `/home/jovyan/v2/lerobot/<dataset>/` |
| Python env | `source /opt/conda/bin/activate lerobot` |

**Before starting, verify the reference is r=28** (an experiment may have changed it):
`python -c "import json; print(json.load(open('/home/jovyan/v2/checker/roi.json'))['pizza'])"` must print
`[118, 131, 28]`. If not, restore `cp /home/jovyan/v2/checker/roi_r28.json /home/jovyan/v2/checker/roi.json`
and `cp /home/jovyan/v2/checker/change_r28.pt /home/jovyan/v2/checker/change.pt` (backups made before the experiment).

Do not modify anything in `/home/jovyan/datasets` or `/home/jovyan/v2/lerobot`. Do not commit data,
images or weights (see `/home/jovyan/.gitignore`); commit scripts and docs only.

## Key facts about the existing code

- Sector geometry (must stay identical to training): for center `(cx, cy)` and radius `r`, quadrant boxes
  are the four quarter squares of the circle, padded by `max(2, int(0.08 * r))` px; `CENTER` is the square
  of half-size `0.45 * r`. Quadrant names (`TL/TR/BR/BL`) map to image quadrants via `roi.json["mapping"]`.
  `PizzaLocator.sector_boxes(cx, cy, r, img_shape)` already implements this exactly.
- `PizzaLocator(ref_image, method="red")` calibrates once on a reference frame (frame 0 of
  `cheese-r1r10-2026-10-24` episode 0 was used) and `locate(img)` returns `(cx, cy, r)` of the sector circle.
- `ChangeChecker.check_step(before, after, target, topping, thr)` returns `(ok, p, elsewhere)`; it uses
  `self.roi`, so assigning `ck.roi = boxes` switches it to dynamic boxes.
- Supported sectors: `TL, TR, BR, BL, CENTER`. `GAP` is not supported. Only **pepperoni** is reliable
  (validation: 99.6% detected, 1 false "added" out of 1206 at thr 0.5; recommended thr 0.65).
  Cheese is not reliable; never use the checker to verify cheese steps.

## What to build

### 1. `/home/jovyan/v2/scripts/failure_checker_v2.py`

```python
class FailureCheckerV2:
    def __init__(self, model="/home/jovyan/v2/checker/change.pt",
                 roi_json="/home/jovyan/v2/checker/roi.json",
                 ref_image=None,            # reference top frame for locator calibration; if None, load
                                            # frame 0 of cheese-r1r10-2026-10-24 episode 0 from /home/jovyan/v2/lerobot
                 radius=28,                 # fixed sector radius (user default)
                 max_offset_px=15,          # located center further than this from roi.json center -> fallback
                 thr=0.65, device=None): ...

    def start(self, top_rgb) -> dict:
        """Call once at the start of a run (arm at home). Locates the pizza with method 'red', builds sector
        boxes with the FIXED radius, assigns them to the change checker. Falls back to fixed roi.json boxes
        if the pizza is not found or is further than max_offset_px from the reference center.
        Returns a status dict (see 'UI data contract')."""

    def check_step(self, before_rgb, after_rgb, target, topping="pepperoni") -> dict:
        """Same decision as v1 check_step, but with the boxes chosen in start(). Returns a result dict."""

    def render(self, top_rgb, result=None) -> "np.ndarray":
        """RGB overlay for the UI: located circle, sector boxes with names, target sector highlighted,
        per-sector probabilities if a result is given."""
```

Implementation notes:
- Use `PizzaLocator(ref_image, method="red")`, take only `(cx, cy)` from `locate()`, radius = `radius` arg.
- Build boxes with `PizzaLocator.sector_boxes(cx, cy, radius, top_rgb.shape)`.
- Reuse `ChangeChecker` for the model; do not copy model code.
- Guard against non-supported targets and toppings: return `{"supported": False, ...}` for cheese or GAP.
- Keep it fast: `start()` < 100 ms, `check_step()` < 100 ms on CPU (5 sectors).

### 2. Runner integration (`/home/jovyan/scripts/run_lerobot_policy.py`)

Add `--failure-checker v2` (default keeps v1 behaviour). With v2: on the first tick of the run call
`start(top)` with the top frame (Station frame image 0, decoded JPEG -> RGB), print the status, and in the
existing retry loop replace the v1 `checker.check_step(...)` call with `FailureCheckerV2.check_step(...)`.
Copy the new module next to the runner (as was done for `change_checker.py`). Test with `--dry-run` first.

### 3. UI integration

The user wants v2 visible in "the UI". **Find the UI code first** (search the home directory and the
NormaCore / robot-pizza-cup checkouts for the app that shows camera frames and run status; if it cannot be
found, ask the user for the path or framework before writing UI code). Then add:

- a toggle or setting to choose checker v1 / v2;
- after `start()`: the top-camera frame with the located pizza circle and the five sector boxes
  (`render()` output), plus status text: found / fallback, offset from reference in px;
- after each checked step: target sector, topping, probability, OK / MISSING, sectors where the topping
  appeared elsewhere, attempt number and retry decision; the overlay with the target sector highlighted.

#### UI data contract (what `start()` / `check_step()` return; JSON-serialisable)

```json
// start()
{"version": "v2", "method": "red", "found": true, "fallback": false,
 "center": [118.4, 130.9], "radius": 28, "offset_px": 0.6,
 "boxes": {"TL": [x0, y0, x1, y1], "TR": [...], "BR": [...], "BL": [...], "CENTER": [...]}}

// check_step()
{"version": "v2", "supported": true, "target": "TL", "topping": "pepperoni",
 "p": 0.97, "ok": true, "threshold": 0.65,
 "per_sector": {"TL": 0.97, "TR": 0.01, "BR": 0.00, "BL": 0.02, "CENTER": 0.01},
 "elsewhere": {}, "boxes_source": "dynamic"}
```

## Validation (must pass before UI work)

1. Offline, held-out rounds: run `python /home/jovyan/v2/scripts/r_eval_tracking.py red` and extend it (or
   write `t_eval_v2.py`) to evaluate `FailureCheckerV2` the same way (locate on frame 0 of each episode,
   fixed radius 28). v2 must be **not worse than `fixed`** on pepperoni: same or higher `detected`, same or
   fewer `false_added`. Report the table.
2. Robustness: shift test frames by a few pixels (e.g. `np.roll` by ±5 px in x and y, both before and after
   frames) to simulate a misplaced pizza. v2 should keep its accuracy; `fixed` is expected to degrade. Report both.
3. Fallback: feed a frame without a pizza (e.g. a blank or heavily cropped image) and check that `start()`
   returns `fallback: true` with the fixed boxes and does not crash.
4. Live: run the robot runner with `--failure-checker v2 --dry-run` and confirm the printed center is within a
   few px of the pizza on the real frame; save the `render()` image for the user.

## Deliverables

- `failure_checker_v2.py`, evaluation script, runner flag, UI changes;
- `/home/jovyan/v2/scripts/FAILURE_CHECKER_V2.md`: short doc in the style of `CHANGE_CHECKER.md` (paths,
  API, data contract, validation numbers, limitations);
- a commit with scripts and docs only.

## Known limitations to keep in mind

- Localization must run on a frame where the arm does not cover the pizza.
- Heavy topping coverage can shift the red-region center; that is why localization runs once at start.
- The change model was trained at r=28 sector scale; a very different pizza size on the image (camera
  moved up or down) is out of distribution and would need retraining with scale augmentation.
