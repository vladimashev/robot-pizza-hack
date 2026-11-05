"""Run a whole challenge on ElRobot: recipe -> steps (topping, sector) -> one SmolVLA prompt per step, with the
failure checker deciding after every pepperoni step whether the step is repeated.

Rebuilt after the laptop runtime (smolvla_py scripts/run_challenge.py + pizza/robot.py + pizza/recipe.py) was
lost; the checker contract follows checker/FAILURE_CHECKER_V2.md. Station and model I/O come from
run_lerobot_policy.py (same dir), so the same env works: lerobot==0.3.3 [smolvla], station_py, opencv.

Run:
  1. connect with the arm at home: its joint positions become the home pose, the top frame goes to
     FailureCheckerV2.start() (pizza located once, sector boxes built).
  2. per step: before = top frame at home -> the policy runs the step prompt until the arm has left home and is
     back (or --max-step-ticks) -> after = top frame -> check_step(); not OK and retries left -> same step again.
  3. no new step is started after --time-limit seconds (challenge timeout: 3 min ch1, 8 min ch2/ch3).

Prompts must match the training `task` strings exactly. The v2 prompts came from the (lost) manifests, so the
default template below is the v1 one: pass --prompts prompts.json ({"pepperoni:TL": "...", ...}) or check the
defaults against a dataset's meta/tasks.jsonl with --tasks-jsonl.

Examples:
  python run_challenge.py ch2 --plan-only
  python run_challenge.py recipe --recipe "Build a pizza: place 4 pepperoni on the right half, then spread cheese" --plan-only
  python run_challenge.py ch2 --checkpoint ~/outputs/smolvla_v2_full/checkpoints/020000/pretrained_model \
      --bus-serial <FOLLOWER_SERIAL> --checker-ref ~/v2/checker/ref_top.png --checker-log logs/run1
  python run_challenge.py ch2 ... --dry-run        # locate the pizza on the live image, the arm is not moved
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "checker"))

QUADS = ("TL", "TR", "BR", "BL")                     # clockwise, as recorded
QUAD_NAMES = {"TL": "top-left", "TR": "top-right", "BR": "bottom-right", "BL": "bottom-left"}
REGIONS = {"left half": ("TL", "BL"), "right half": ("TR", "BR"),
           "top half": ("TL", "TR"), "bottom half": ("BL", "BR")}
TOPPINGS = {"cheese": "cheese", "pepperoni": "pepperoni", "salami": "pepperoni"}
SPREAD = ("cheese", "other-spread")                  # one step per sector, not a piece count
NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
PROMPT = "put the {topping} on the {quadrant} quadrant of the pizza"
PRESETS = {
    "ch1-cheese": "spread the cheese over the pizza",
    "ch1-pepperoni": "place 6 pepperoni pieces on the pizza",
    "ch2": "Build a pizza: place and spread cheese over the pizza, then 6 pepperoni pieces",
    "ch3a": "Build a pizza: place and spread cheese over the pizza, then 6 pepperoni pieces",
}


# ---- recipe -> steps -> prompts ----

def parse_recipe(text: str, spread_per_sector: int = 1) -> list[tuple[str, str]]:
    """'Build a pizza: 4 pepperoni on the right half, then spread cheese' -> [(topping, sector), ...].
    Known toppings by name; any other ingredient with a count is 'other' (pieces), without a count but with
    spread/sprinkle it is 'other-spread'. Pieces cycle over the region's sectors, spreads visit each sector."""
    steps = []
    body = text.split(":", 1)[-1].lower()
    for part in re.split(r",|;|\bthen\b|\band after that\b|\bafter that\b", body):
        part = part.strip()
        if not part:
            continue
        words = re.findall(r"[a-z]+|\d+", part)
        topping = next((TOPPINGS[w] for w in words if w in TOPPINGS), None)
        count = next((int(w) if w.isdigit() else NUMBERS[w] for w in words if w.isdigit() or w in NUMBERS), None)
        if topping is None:
            if count:
                topping = "other"
            elif re.search(r"spread|sprinkle|scatter", part):
                topping = "other-spread"
            else:
                print(f"recipe: skipped {part!r} (no ingredient)")
                continue
        sectors = next((q for k, q in REGIONS.items() if k in part), QUADS)
        if topping in SPREAD:
            steps += [(topping, s) for s in sectors for _ in range(spread_per_sector)]
        else:
            steps += [(topping, sectors[i % len(sectors)]) for i in range(count or len(sectors))]
    return steps


def parse_steps(spec: str) -> list[tuple[str, str]]:
    """'cheese:TL,pepperoni:TR' -> [('cheese', 'TL'), ('pepperoni', 'TR')]."""
    return [tuple(x.strip().split(":")) for x in spec.split(",") if x.strip()]


def prompt_for(topping: str, sector: str, prompts: dict) -> str:
    key = f"{topping}:{sector}"
    if key in prompts:
        return prompts[key]
    if topping in ("cheese", "pepperoni") and sector in QUAD_NAMES:
        return PROMPT.format(topping=topping, quadrant=QUAD_NAMES[sector])
    raise SystemExit(f"no prompt for {key}: add it to --prompts")


# ---- logging ----

class EventLog:
    """events.jsonl (start, step, check, decision, end) + overlay PNGs in --checker-log."""

    def __init__(self, root: Path | None):
        self.root = root
        if root:
            root.mkdir(parents=True, exist_ok=True)
            self.f = open(root / "events.jsonl", "a")

    def __call__(self, event: str, **kw):
        rec = {"t": round(time.time(), 3), "event": event, **kw}
        print(json.dumps(rec, default=str))
        if self.root:
            self.f.write(json.dumps(rec, default=str) + "\n")
            self.f.flush()

    def image(self, name: str, rgb):
        if self.root and rgb is not None:
            Image.fromarray(np.asarray(rgb, np.uint8)).save(self.root / name)


# ---- robot ----

class Robot:
    """Station connection + policy. One step = run one prompt from home back to home."""

    def __init__(self, args):
        from run_lerobot_policy import import_station, load_policy
        self.args = args
        (self.new_client, self.send_commands, self.normvla, self.st3215,
         self.commands, self.drivers) = import_station(args.norma_root)
        self.motor_ids = [int(x) for x in args.motor_ids.split(",")]
        self.image_keys = tuple(args.image_keys.split(","))
        self.zero = [int(x) for x in args.zero_state_dims.split(",") if x != ""] if args.zero_state_dims else []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = None
        if not args.dry_run:
            print(f"loading {args.checkpoint} on {self.device} ...")
            self.policy = load_policy(args.checkpoint, args.n_action_steps, self.device)
            if self.policy.config.type != "smolvla":
                raise SystemExit("run_challenge needs a prompt-conditioned policy (SmolVLA)")
        self.client, self.last_id, self.home = None, b"", None

    async def connect(self):
        self.client = await self.new_client(self.args.server, logging.getLogger("run_challenge"))
        frame = await self.frame()
        self.home = self.state(frame)
        return frame

    async def frame(self):
        from run_lerobot_policy import QUEUE_ID
        while True:
            qr = self.client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
            entry = await asyncio.wait_for(qr.data.get(), timeout=self.args.timeout)
            if entry is None:
                raise RuntimeError(f"{QUEUE_ID} closed without a frame ({qr.err})")
            frame = self.normvla.FrameReader(memoryview(bytes(entry.Data)))
            fid = bytes(frame.get_global_frame_id())
            if fid != self.last_id:
                self.last_id = fid
                return frame
            await asyncio.sleep(0.02)

    @staticmethod
    def state(frame) -> np.ndarray:
        return np.array([j.get_position_norm() for j in frame.get_joints() or []], np.float32)

    def top(self, frame, size=None) -> np.ndarray:
        """cam0 (top) as RGB, resized to the checker's image size if it differs."""
        with Image.open(io.BytesIO(bytes(frame.get_images()[0].get_jpeg()))) as im:
            im = im.convert("RGB")
            if size and im.size != tuple(size):
                im = im.resize(tuple(size), Image.BILINEAR)
            return np.asarray(im)

    async def settle(self):
        await asyncio.sleep(self.args.settle)
        return await self.frame()

    def _sync_write(self, raws):
        from run_lerobot_policy import ST3215_TARGET_POS_REGISTER
        st = self.st3215
        motors = [st.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=r.to_bytes(2, "little"))
                  for m, r in zip(self.motor_ids, raws)]
        sync = st.ST3215SyncWriteCommand(address=ST3215_TARGET_POS_REGISTER, motors=motors)
        cmd = st.Command(target_bus_serial=self.args.bus_serial, sync_write=sync)
        return self.commands.DriverCommand(type=self.drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())

    @torch.no_grad()
    async def run_step(self, prompt: str) -> dict:
        """Ticks until the arm has left home (> --leave-tol) and stayed back within --home-tol for --home-ticks."""
        from run_lerobot_policy import frame_to_batch
        a = self.args
        self.policy.reset()
        left, still, skipped = False, 0, 0
        for n in range(1, a.max_step_ticks + 1):
            frame = await self.frame()
            dist = float(np.abs(self.state(frame) - self.home).max())
            left |= dist > a.leave_tol
            still = still + 1 if left and dist < a.home_tol else 0
            if still >= a.home_ticks:
                return {"status": "done", "ticks": n, "skipped": skipped}
            batch, ranges = frame_to_batch(frame, self.image_keys, prompt, "smolvla", self.device)
            if self.zero:
                batch["observation.state"][:, self.zero] = 0.0
            goal = self.policy.select_action(batch)[0].float().cpu().clamp(0.0, 1.0).numpy()
            raws = [max(lo, min(hi, int(round(lo + float(g) * (hi - lo))))) for g, (lo, hi) in zip(goal, ranges)]
            max_d = max(abs(r - int(j.get_position())) for r, j in zip(raws, frame.get_joints()))
            if a.max_delta_ticks and max_d > a.max_delta_ticks:
                skipped += 1
                continue
            await self.send_commands(self.client, [self._sync_write(raws)])
        return {"status": "timeout", "ticks": a.max_step_ticks, "skipped": skipped, "left_home": left}


# ---- checker ----

def make_checker(args):
    if args.failure_checker == "off":
        return None
    from failure_checker_v2 import FailureCheckerV2
    d = Path(args.checker_dir).expanduser()
    ref = str(Path(args.checker_ref).expanduser()) if args.checker_ref else None
    fc = FailureCheckerV2(model=str(d / "change.pt"), roi_json=str(d / "roi.json"), ref_image=ref,
                          max_offset_px=args.checker_max_offset, thr=args.checker_thr)
    fc.image_size = json.load(open(d / "roi.json"))["image_size"]
    return fc


# ---- run ----

async def run(args, steps, prompts, robot, checker, log):
    frame = await robot.connect()
    size = getattr(checker, "image_size", None)
    start_img = robot.top(frame, size)
    if checker is None:
        log("start", checker="off")
    else:
        if args.failure_checker == "v1":
            st = checker.use_fixed("--failure-checker v1")
        elif args.checker_ref:
            st = checker.start(start_img)
        else:
            st = checker.use_fixed("no --checker-ref: v2 needs the locator's reference frame")
        log("start", checker=args.failure_checker, **{k: v for k, v in st.items() if k != "boxes"})
        log.image("start.png", checker.render(start_img))
    if args.dry_run:
        log("end", status="dry run, arm not moved")
        return

    t0, results = time.monotonic(), []
    before = start_img
    for i, (topping, sector) in enumerate(steps, 1):
        prompt = prompts[i - 1]
        for attempt in range(1, args.retries + 2):
            if time.monotonic() - t0 > args.time_limit:
                log("end", status="time limit", elapsed=round(time.monotonic() - t0, 1), done=results)
                return results
            log("step", i=i, n=len(steps), topping=topping, sector=sector, attempt=attempt, prompt=prompt)
            res = await robot.run_step(prompt)
            after = robot.top(await robot.settle(), size)
            ok = None
            if checker is not None and res["status"] == "done":
                chk = checker.check_step(before, after, sector, topping)
                log("check", i=i, attempt=attempt, **chk)
                log.image(f"step{i:02d}_try{attempt}.png", checker.render(after, chk))
                ok = chk["ok"]
            retry = ok is False and attempt <= args.retries
            log("decision", i=i, attempt=attempt, run=res, ok=ok,
                action="retry" if retry else ("gave up" if ok is False else "next"))
            before = after
            if res["status"] == "timeout" and not res.get("left_home", True):
                log("end", status="arm never left home, stopping", i=i)
                return results
            if not retry:
                results.append(dict(i=i, topping=topping, sector=sector, ok=ok, attempts=attempt))
                break
    log("end", status="done", elapsed=round(time.monotonic() - t0, 1), done=results)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("challenge", choices=[*PRESETS, "recipe", "steps"])
    p.add_argument("--recipe", help="Recipe text for `recipe` (e.g. the Mystery Pizza instruction).")
    p.add_argument("--steps", help="Explicit steps for `steps`: cheese:TL,pepperoni:TR,...")
    p.add_argument("--spread-per-sector", type=int, default=1, help="Cheese/spread steps per sector.")
    p.add_argument("--prompts", type=Path, help='JSON {"topping:SECTOR": prompt}; overrides the template.')
    p.add_argument("--tasks-jsonl", type=Path, help="Training meta/tasks.jsonl: fail if a prompt is not in it.")
    p.add_argument("--plan-only", action="store_true", help="Print steps and prompts, do not connect.")
    p.add_argument("--dry-run", action="store_true", help="Connect and locate the pizza, do not move the arm.")
    p.add_argument("--checkpoint", type=Path, help="LeRobot pretrained_model dir.")
    p.add_argument("--server", default="localhost")
    p.add_argument("--bus-serial", help="Follower ST3215 bus id from the Station UI.")
    p.add_argument("--motor-ids", default="1,2,3,4,5,6,7,8")
    p.add_argument("--image-keys", default="observation.images.top,observation.images.front")
    p.add_argument("--zero-state-dims", default="",
                   help="State dims zeroed at train time (c_compat variant): '7' nogripper, '0,...,7' nostate.")
    p.add_argument("--n-action-steps", type=int, default=0, help="Ticks per chunk before re-planning; 0 = checkpoint.")
    p.add_argument("--max-delta-ticks", type=int, default=200)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--max-step-ticks", type=int, default=300, help="Per step (data is 10 Hz: 300 = ~30 s).")
    p.add_argument("--leave-tol", type=float, default=0.10, help="Max joint distance (norm) that counts as away.")
    p.add_argument("--home-tol", type=float, default=0.04, help="Max joint distance (norm) that counts as home.")
    p.add_argument("--home-ticks", type=int, default=5, help="Ticks at home that end a step.")
    p.add_argument("--settle", type=float, default=0.5, help="Seconds before the after frame.")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--time-limit", type=float, default=480, help="Seconds; no new step after this.")
    p.add_argument("--failure-checker", choices=["v2", "v1", "off"], default="v2")
    p.add_argument("--checker-dir", default="~/v2/checker", help="Holds change.pt and roi.json.")
    p.add_argument("--checker-ref", help="Locator reference frame (ref_top.png); without it v2 = fixed boxes.")
    p.add_argument("--checker-max-offset", type=float, default=15)
    p.add_argument("--checker-thr", type=float, default=0.65)
    p.add_argument("--checker-log", type=Path, help="Dir for events.jsonl and overlay PNGs.")
    p.add_argument("--norma-root", type=Path, default=Path("~/norma-core").expanduser())
    return p.parse_args(argv)


def build_plan(args):
    if args.challenge == "steps":
        if not args.steps:
            raise SystemExit("`steps` needs --steps")
        steps = parse_steps(args.steps)
    else:
        text = args.recipe if args.challenge == "recipe" else PRESETS[args.challenge]
        if not text:
            raise SystemExit("`recipe` needs --recipe")
        steps = parse_recipe(text, args.spread_per_sector)
    table = json.load(open(args.prompts)) if args.prompts else {}
    prompts = [prompt_for(t, s, table) for t, s in steps]
    if args.tasks_jsonl:
        known = {json.loads(l)["task"] for l in open(args.tasks_jsonl) if l.strip()}
        unknown = sorted(set(prompts) - known)
        if unknown:
            raise SystemExit("prompts not in training tasks:\n  " + "\n  ".join(unknown))
    return steps, prompts


def main(argv=None):
    args = parse_args(argv)
    steps, prompts = build_plan(args)
    print(f"{len(steps)} steps:")
    for i, ((t, s), pr) in enumerate(zip(steps, prompts), 1):
        print(f"  {i:2d}. {t:12s} {s:6s} {pr!r}")
    if args.plan_only:
        return
    if not args.bus_serial or (not args.dry_run and not args.checkpoint):
        raise SystemExit("--bus-serial and --checkpoint are required (except --plan-only; --dry-run needs no checkpoint)")
    log = EventLog(args.checker_log)
    try:
        asyncio.run(run(args, steps, prompts, Robot(args), make_checker(args), log))
    except KeyboardInterrupt:
        log("end", status="interrupted")


if __name__ == "__main__":
    main()
