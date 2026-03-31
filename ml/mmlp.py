import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, SWALR
import os
import json
import math
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

label_map = {
    "angry": 0, "fear": 1, "happy": 2,
    "neutral": 3, "sad": 4, "surprise": 5
}
idx_to_label = {v: k for k, v in label_map.items()}

# ===================== PREPROCESS =====================
LANDMARK_GROUPS = {
    "left_eye":   [33, 160, 158, 133, 153, 144],
    "right_eye":  [362, 385, 387, 263, 373, 380],
    "mouth":      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "left_brow":  [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "nose":       [1, 2, 4, 5, 6],
}

def group_centroid(coords, indices):
    xs = [coords[i][0] for i in indices]
    ys = [coords[i][1] for i in indices]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def preprocess(landmarks):
    ref = landmarks[1]
    centered = [[x - ref[0], y - ref[1]] for x, y, z in landmarks]
    left_eye  = centered[33]
    right_eye = centered[263]
    scale = math.sqrt((left_eye[0]-right_eye[0])**2 + (left_eye[1]-right_eye[1])**2) + 1e-8
    scaled = [[x/scale, y/scale] for x, y in centered]

    features = []
    for x, y in scaled:
        dist  = math.sqrt(x**2 + y**2)
        angle = math.atan2(y, x)
        features.extend([x, y, dist, angle])

    centroids = {n: group_centroid(scaled, idx) for n, idx in LANDMARK_GROUPS.items()}
    group_names = list(LANDMARK_GROUPS.keys())
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            ax, ay = centroids[group_names[i]]
            bx, by = centroids[group_names[j]]
            features.append(math.sqrt((ax-bx)**2 + (ay-by)**2))

    def ear(pts):
        p = [scaled[i] for i in pts]
        v1 = math.sqrt((p[1][0]-p[5][0])**2+(p[1][1]-p[5][1])**2)
        v2 = math.sqrt((p[2][0]-p[4][0])**2+(p[2][1]-p[4][1])**2)
        h  = math.sqrt((p[0][0]-p[3][0])**2+(p[0][1]-p[3][1])**2)+1e-8
        return (v1+v2)/(2.0*h)

    def mar():
        top, bot = scaled[13], scaled[14]
        l, r     = scaled[61], scaled[291]
        v = math.sqrt((top[0]-bot[0])**2+(top[1]-bot[1])**2)
        h = math.sqrt((l[0]-r[0])**2+(l[1]-r[1])**2)+1e-8
        return v/h

    left_ear  = ear(LANDMARK_GROUPS["left_eye"])
    right_ear = ear(LANDMARK_GROUPS["right_eye"])
    features.extend([left_ear, right_ear, (left_ear+right_ear)/2, mar()])

    lbx,lby = centroids["left_brow"];  lex,ley = centroids["left_eye"]
    rbx,rby = centroids["right_brow"]; rex,rey = centroids["right_eye"]
    lr = math.sqrt((lbx-lex)**2+(lby-ley)**2)
    rr = math.sqrt((rbx-rex)**2+(rby-rey)**2)
    features.extend([lr, rr, (lr+rr)/2])

    x = torch.tensor(features, dtype=torch.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x


# ===================== DATASET =====================
class FacialDataSet(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_name in os.listdir(root_dir):
            if label_name not in label_map:
                continue
            label_path = os.path.join(root_dir, label_name)
            for file in os.listdir(label_path):
                try:
                    with open(os.path.join(label_path, file), "r") as f:
                        data = json.load(f)
                        x = preprocess(data["landmarks"])
                        y = label_map[label_name]
                        self.samples.append((x, y))
                except:
                    continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


class SplitDataset(Dataset):
    def __init__(self, base, indices, is_train=False):
        self.base     = base
        self.indices  = indices
        self.is_train = is_train

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base.samples[self.indices[idx]]

        if self.is_train:
            if random.random() < 0.7:
                x = x + torch.randn_like(x) * 0.02
            if random.random() < 0.5:
                x = x * (1 + torch.randn(1).item() * 0.01)
            if random.random() < 0.3:
                mask = torch.rand_like(x) > 0.08
                x = x * mask

        return x, torch.tensor(y, dtype=torch.long)


# ===================== MIXUP / CUTMIX =====================
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def cutmix_1d(x, y, alpha=1.0):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0), device=x.device)
    n     = x.size(1)
    cut   = int(n * (1 - lam))
    start = random.randint(0, n - cut)
    x_new = x.clone()
    x_new[:, start:start+cut] = x[idx, start:start+cut]
    return x_new, y, y[idx], lam

def mixed_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1-lam) * criterion(pred, yb)


# ===================== MODEL =====================
class StochasticDepthResBlock(nn.Module):
    def __init__(self, size, dropout=0.2, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.act = nn.GELU()

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_prob:
            return self.act(x)
        return self.act(x + self.block(x))


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, drop_prob=0.1):
        super().__init__()
        H = 384

        self.stem = nn.Sequential(
            nn.Linear(input_size, H),
            nn.BatchNorm1d(H),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.res1 = StochasticDepthResBlock(H, dropout=0.2, drop_prob=drop_prob)
        self.res2 = StochasticDepthResBlock(H, dropout=0.2, drop_prob=drop_prob*1.5)
        self.res3 = StochasticDepthResBlock(H, dropout=0.2, drop_prob=drop_prob*2.0)

        self.mid = nn.Sequential(
            nn.Linear(H, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.res4 = StochasticDepthResBlock(192, dropout=0.15, drop_prob=drop_prob)

        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.mid(x)
        x = self.res4(x)
        return self.head(x)


# ===================== TRAIN =====================
def train():
    dataset_path = "../my_react_app/public/train3_landmarks"

    print("Loading dataset...")
    full_dataset = FacialDataSet(dataset_path)
    print(f"Loaded {len(full_dataset)} samples")

    labels = [y for _, y in full_dataset.samples]
    print("Class distribution:", Counter(labels))

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    input_size    = len(full_dataset.samples[0][0])
    train_dataset = SplitDataset(full_dataset, train_idx, is_train=True)
    val_dataset   = SplitDataset(full_dataset, val_idx,   is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)

    label_counts  = Counter(labels)
    total         = len(labels)
    class_weights = torch.tensor([
        total / (len(label_map) * label_counts[i])
        for i in range(len(label_map))
    ], dtype=torch.float32).to(device)
    print("Class weights:", class_weights.tolist())

    model = MLP(input_size, len(label_map), drop_prob=0.12).to(device)
    print(f"Input size: {input_size}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # FIX 1: Start at full LR immediately — no warmup needed for AdamW on this data size.
    # FIX 2: Higher base LR (3e-4 proven to work in v1/v2).
    # FIX 3: Moderate weight decay (not too high — 5e-4 was hurting convergence).
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)

    NUM_EPOCHS = 100
    SWA_START  = 65

    # FIX 4: Simple cosine schedule from epoch 1, no warmup.
    # Warmup was causing the slow start (epoch 1 at 17% train acc).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SWA_START, eta_min=1e-5
    )

    swa_model     = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=10)
    swa_started   = False

    best_val_acc = 0.0
    patience     = 20
    wait         = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_correct = 0
        train_total   = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            r = random.random()
            if r < 0.4:
                x_m, ya, yb, lam = mixup(x, y, alpha=0.4)
                out  = model(x_m)
                loss = mixed_criterion(criterion, out, ya, yb, lam)
                with torch.no_grad():
                    pred = model(x).argmax(dim=1)
            elif r < 0.65:
                x_m, ya, yb, lam = cutmix_1d(x, y, alpha=1.0)
                out  = model(x_m)
                loss = mixed_criterion(criterion, out, ya, yb, lam)
                with torch.no_grad():
                    pred = model(x).argmax(dim=1)
            else:
                out  = model(x)
                loss = criterion(out, y)
                pred = out.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_correct += (pred == y).sum().item()
            train_total   += y.size(0)

        train_acc = train_correct / train_total

        if epoch >= SWA_START:
            if not swa_started:
                print(f"  >> SWA started at epoch {epoch+1}")
                swa_started = True
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total   += y.size(0)
        val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch < SWA_START:
                print(f"Early stopping | Best val: {best_val_acc:.4f}")
                # Don't actually break — continue into SWA phase
                # which often rescues a plateau
                if epoch < SWA_START - 5:
                    break

    # ---- SWA evaluation ----
    if swa_started:
        print("\nUpdating SWA batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        swa_correct = 0
        swa_total   = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = swa_model(x).argmax(dim=1)
                swa_correct += (pred == y).sum().item()
                swa_total   += y.size(0)
        swa_acc = swa_correct / swa_total
        print(f"SWA Val Acc: {swa_acc:.4f}  (base best: {best_val_acc:.4f})")

        if swa_acc > best_val_acc:
            print("SWA model is better — saving.")
            torch.save(swa_model.module.state_dict(), "best_model.pth")
            best_val_acc = swa_acc

    # ---- Per-class breakdown ----
    print("\nPer-class accuracy on val set:")
    best_state = torch.load("best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(best_state)
    model.eval()
    class_correct = Counter()
    class_total   = Counter()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            for p, t in zip(pred.cpu(), y.cpu()):
                class_total[t.item()]   += 1
                class_correct[t.item()] += int(p == t)

    for cls_idx in range(len(label_map)):
        acc = class_correct[cls_idx] / max(class_total[cls_idx], 1)
        bar = "█" * int(acc * 20)
        print(f"  {idx_to_label[cls_idx]:>8}: {acc:.3f} {bar}  ({class_total[cls_idx]} samples)")

    print(f"\nDone! Best val: {best_val_acc:.4f}")
    torch.save(model.state_dict(), "expression.pth")


# ===================== ONNX EXPORT =====================
def export_onnx():
    dataset_path = "../my_react_app/public/train3_landmarks"
    input_size   = None
    for label_name in os.listdir(dataset_path):
        if label_name not in label_map: continue
        for file in os.listdir(os.path.join(dataset_path, label_name)):
            try:
                with open(os.path.join(dataset_path, label_name, file)) as f:
                    input_size = len(preprocess(json.load(f)["landmarks"]))
                    break
            except: continue
        if input_size: break

    print(f"Exporting with input_size={input_size}")
    model = MLP(input_size, len(label_map))
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, input_size)
    torch.onnx.export(
        model, dummy, "expression.onnx",
        input_names=["landmarks"], output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print("Exported to expression.onnx")


if __name__ == "__main__":
    train()
    export_onnx()