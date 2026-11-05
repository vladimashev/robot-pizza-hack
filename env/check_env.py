"""Smoke test of the env: GPU, lerobot policies, video decode, repo modules, Station client.
    python env/check_env.py [--norma-root ~/norma-core]"""
import argparse, os, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--norma-root", type=Path, default=Path(os.environ.get("NORMA_CORE_ROOT", "~/norma-core")).expanduser())
args = p.parse_args()
fails = []


def check(name, fn):
    try:
        print(f"ok    {name}: {fn() or ''}")
    except Exception as e:
        fails.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__}: CUDA not available")
    x = torch.randn(256, 256, device="cuda")
    (x @ x).sum().item()
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def policies():
    import lerobot
    from lerobot.policies.factory import get_policy_class
    return f"lerobot {lerobot.__version__}: {get_policy_class('smolvla').__name__}, {get_policy_class('act').__name__}"


def video():
    import av, numpy as np
    from lerobot.datasets.video_utils import decode_video_frames
    path = os.path.join(tempfile.mkdtemp(), "t.mp4")
    with av.open(path, "w") as c:                       # libx264, as the exporter writes
        s = c.add_stream("libx264", rate=10)
        s.width = s.height = 64
        s.pix_fmt = "yuv420p"
        for k in range(10):
            for pk in s.encode(av.VideoFrame.from_ndarray(np.full((64, 64, 3), 20 * k, np.uint8), format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    fr = decode_video_frames(path, [0.0, 0.5], 0.05, backend="pyav")
    return f"pyav {av.__version__}, decoded {tuple(fr.shape)}"


def repo():
    sys.path[:0] = [str(REPO / "checker"), str(REPO / "robot")]
    import failure_checker_v2, topping_checker, run_challenge  # noqa: F401
    return "checker/, robot/"


def station():
    sys.path.insert(0, str(REPO / "robot"))
    from run_lerobot_policy import import_station
    import_station(args.norma_root)
    return str(args.norma_root)


for name, fn in [("gpu", gpu), ("lerobot policies", policies), ("video decode", video),
                 ("repo modules", repo), ("station client", station)]:
    check(name, fn)
if fails:
    sys.exit(f"\n{len(fails)} check(s) failed: {', '.join(fails)}")
print("\nall checks passed")
