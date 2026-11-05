"""Train the topping checker (ResNet18, two sigmoid outputs: has_cheese, has_pepperoni).
Validation = whole rounds 5 and 10 held out. Saves ~/v2/checker/checker.pt."""
import json, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

H = os.path.expanduser("~")
OUT = f"{H}/v2/checker"
EPOCHS, BS, LR, CROP = 12, 64, 3e-4, 144
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{OUT}/labels.csv")
df["val"] = df["round"].fillna(0).astype(int) % 5 == 0
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF_TRAIN = T.Compose([T.RandomResizedCrop(CROP, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
                      T.RandomRotation(8), T.ColorJitter(0.3, 0.3, 0.3, 0.03), T.ToTensor(), norm])
TF_VAL = T.Compose([T.CenterCrop(CROP), T.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, d, tf):
        self.d, self.tf = d.reset_index(drop=True), tf

    def __len__(self):
        return len(self.d)

    def __getitem__(self, k):
        r = self.d.iloc[k]
        return self.tf(Image.open(r.path).convert("RGB")), torch.tensor([r.has_cheese, r.has_pepperoni], dtype=torch.float32)


tr = DataLoader(DS(df[~df.val], TF_TRAIN), BS, shuffle=True, num_workers=4)
va = DataLoader(DS(df[df.val], TF_VAL), BS, num_workers=4)
print(f"train {(~df.val).sum()} crops, val {df.val.sum()} crops ({df[df.val].ep.nunique()} episodes)")

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(512, 2)
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
lossf = nn.BCEWithLogitsLoss()


def evaluate():
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in va:
            P.append(torch.sigmoid(model(x.to(dev))).cpu())
            Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for x, y in tr:
        loss = lossf(model(x.to(dev)), y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x)
    sched.step()
    p, y = evaluate()
    acc = ((p > 0.5) == y).mean(0)
    print(f"epoch {ep + 1:2d}  train loss {tot / len(tr.dataset):.3f}  val acc cheese {acc[0]:.3f}  pepperoni {acc[1]:.3f}")

p, y = evaluate()
v = df[df.val].reset_index(drop=True)
v["p_cheese"], v["p_pepperoni"] = p[:, 0], p[:, 1]
v["ok"] = ((p > 0.5) == y).all(1)
print("\nval accuracy by dataset / kind / target:")
print(v.groupby(["src", "kind"]).ok.mean().round(3).to_string())
print(v.groupby("target").ok.mean().round(3).to_string())
print("\nworst val episodes:")
print(v.groupby(["src", "ep", "kind"]).ok.mean().sort_values().head(10).round(2).to_string())
v.to_csv(f"{OUT}/val_predictions.csv", index=False)
roi = json.load(open(f"{OUT}/roi.json"))
torch.save({"state_dict": model.state_dict(), "roi": roi["roi"], "image_size": roi["image_size"],
            "labels": ["cheese", "pepperoni"], "size": 160, "crop": CROP}, f"{OUT}/checker.pt")
print(f"\nsaved {OUT}/checker.pt")
