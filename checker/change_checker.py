"""Runtime change checker: was a topping ADDED to a sector between two top-camera frames?
    ck = ChangeChecker("~/v2/checker/change.pt")
    before = top_frame()            # arm at home, BEFORE running the step
    ... run policy ...
    after = top_frame()             # arm back at home
    ok, p, elsewhere = ck.check_step(before, after, "TL", "pepperoni")
    # ok: added in target; elsewhere: other sectors where it was added (misplacement)"""
import os, sys
import numpy as np, torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h_train_change import build_model, to_input


class ChangeChecker:
    def __init__(self, path, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(os.path.expanduser(path), map_location=self.dev)
        self.roi, self.labels, self.size, self.crop = ck["roi"], ck["labels"], ck["size"], ck["crop"]
        self.model = build_model()
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(self.dev).eval()

    def _crop(self, img, sector):
        im = Image.fromarray(np.asarray(img)).crop(self.roi[sector]).resize((self.size, self.size), Image.BICUBIC)
        return TF.center_crop(TF.to_tensor(im), [self.crop, self.crop])

    @torch.no_grad()
    def added(self, before, after, sector):
        x = to_input(self._crop(before, sector), self._crop(after, sector)).unsqueeze(0).to(self.dev)
        return dict(zip(self.labels, torch.sigmoid(self.model(x))[0].cpu().tolist()))

    def scan(self, before, after):
        return {s: self.added(before, after, s) for s in self.roi}

    def check_step(self, before, after, target, topping, thr=0.5):
        res = self.scan(before, after)
        p = res[target][topping] if target in res else float("nan")
        elsewhere = {s: round(v[topping], 3) for s, v in res.items() if s != target and v[topping] > thr}
        return (p > thr), p, elsewhere
