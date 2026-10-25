"""Runtime topping checker: does the target sector contain the topping?
    ck = ToppingChecker("~/v2/checker/checker.pt")
    ck.done(top_rgb, "TL", "pepperoni")   -> (bool, prob)
    ck.scan(top_rgb)                      -> {"TL": {"cheese": p, "pepperoni": p}, ...}"""
import os
import numpy as np, torch, torch.nn as nn, torchvision
import torchvision.transforms as T
from PIL import Image


class ToppingChecker:
    def __init__(self, path, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(os.path.expanduser(path), map_location=self.dev)
        self.roi, self.labels, self.size = ck["roi"], ck["labels"], ck["size"]
        m = torchvision.models.resnet18()
        m.fc = nn.Linear(512, len(self.labels))
        m.load_state_dict(ck["state_dict"])
        self.model = m.to(self.dev).eval()
        self.tf = T.Compose([T.CenterCrop(ck["crop"]), T.ToTensor(),
                             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def probs(self, img_rgb, target):
        crop = Image.fromarray(np.asarray(img_rgb)).crop(self.roi[target]).resize((self.size, self.size))
        p = torch.sigmoid(self.model(self.tf(crop).unsqueeze(0).to(self.dev)))[0].cpu().tolist()
        return dict(zip(self.labels, p))

    def done(self, img_rgb, target, topping, thr=0.5):
        p = self.probs(img_rgb, target)[topping]
        return p >= thr, p

    def scan(self, img_rgb):
        return {t: self.probs(img_rgb, t) for t in self.roi}
