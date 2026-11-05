"""Train the change checker: ResNet18 with 6-channel input [after, after-before] -> added_cheese, added_pepperoni.
Validation = rounds 5 and 10 held out. Saves ~/v2/checker/change.pt."""
import json, os, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

H = os.path.expanduser("~")
OUT = f"{H}/v2/checker"
EPOCHS, BS, LR, CROP = 15, 64, 3e-4, 144
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DIFF_SCALE = 0.25
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)


def build_model():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    w = m.conv1.weight.data
    conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
    conv.weight.data = torch.cat([w, w], 1) * 0.5
    m.conv1 = conv
    m.fc = nn.Linear(512, 2)
    return m


def to_input(b, a):
    """b, a: float tensors (3,H,W) in [0,1] -> (6,H,W)."""
    return torch.cat([(a - MEAN) / STD, (a - b) / DIFF_SCALE], 0)


class Pairs(Dataset):
    def __init__(self, d, train):
        self.d, self.train = d.reset_index(drop=True), train
        self.jit = T.ColorJitter(0.15, 0.15, 0.1, 0.01)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, k):
        r = self.d.iloc[k]
        b, a = Image.open(r.before).convert("RGB"), Image.open(r.after).convert("RGB")
        if self.train:
            g = T.ColorJitter(0.2, 0.2, 0.15, 0.02)       # shared global look change
            seed = random.randint(0, 10**9)
            torch.manual_seed(seed); b = g(b)
            torch.manual_seed(seed); a = g(a)
            b, a = self.jit(b), self.jit(a)                # small independent lighting change
        x = torch.stack([TF.to_tensor(b), TF.to_tensor(a)])  # (2,3,H,W)
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(x[0], scale=(0.8, 1.0), ratio=(0.9, 1.1))
            x = TF.resized_crop(x, i, j, h, w, [CROP, CROP], antialias=True)
            x = TF.rotate(x, random.uniform(-8, 8))
        else:
            x = TF.center_crop(x, [CROP, CROP])
        y = torch.tensor([r.added_cheese, r.added_pepperoni], dtype=torch.float32)
        return to_input(x[0], x[1]), y


if __name__ == "__main__":
    df = pd.read_csv(f"{OUT}/pairs.csv")
    df["val"] = df["round"].fillna(0).astype(int) % 5 == 0
    tr = DataLoader(Pairs(df[~df.val], True), BS, shuffle=True, num_workers=4)
    va = DataLoader(Pairs(df[df.val], False), BS, num_workers=4)
    print(f"train {(~df.val).sum()} pairs, val {df.val.sum()} pairs ({df[df.val].groupby(['src', 'ep']).ngroups} episodes)")

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    lossf = nn.BCEWithLogitsLoss()

    def evaluate():
        model.eval()
        P = []
        with torch.no_grad():
            for x, _ in va:
                P.append(torch.sigmoid(model(x.to(dev))).cpu())
        return torch.cat(P).numpy()

    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, y in tr:
            loss = lossf(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        sched.step()
        p = evaluate()
        y = df[df.val][["added_cheese", "added_pepperoni"]].values
        acc = ((p > 0.5) == y).mean(0)
        print(f"epoch {ep + 1:2d}  loss {tot / len(tr.dataset):.3f}  val acc cheese {acc[0]:.3f}  pepperoni {acc[1]:.3f}")

    p = evaluate()
    v = df[df.val].reset_index(drop=True)
    v["p_cheese"], v["p_pepperoni"] = p[:, 0], p[:, 1]
    v["ok"] = ((p > 0.5) == v[["added_cheese", "added_pepperoni"]].values).all(1)
    print("\nval accuracy by dataset / pair kind:")
    print(v.groupby(["src", "kind"]).ok.mean().round(3).unstack().to_string())
    print("\nval accuracy by sector:")
    print(v.groupby("sector").ok.mean().round(3).to_string())
    for name, col in (("cheese", "added_cheese"), ("pepperoni", "added_pepperoni")):
        pr = v[f"p_{name}"]
        for thr in (0.3, 0.5, 0.7, 0.9):
            fd = ((pr > thr) & (v[col] == 0)).sum()
            fm = ((pr <= thr) & (v[col] == 1)).sum()
            print(f"{name:9} thr {thr}: false 'added' {fd:4d} / {(v[col] == 0).sum()}   missed {fm:4d} / {(v[col] == 1).sum()}")
    v.to_csv(f"{OUT}/change_val_predictions.csv", index=False)
    roi = json.load(open(f"{OUT}/roi.json"))
    torch.save({"state_dict": model.state_dict(), "roi": roi["roi"], "image_size": roi["image_size"],
                "labels": ["cheese", "pepperoni"], "size": 160, "crop": CROP, "diff_scale": DIFF_SCALE},
               f"{OUT}/change.pt")
    print(f"\nsaved {OUT}/change.pt")
