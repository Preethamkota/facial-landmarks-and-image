import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm
from torch_geometric.nn import knn_graph
from torch_geometric.utils import add_self_loops

import os
import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================== LABEL MAP =====================
label_map = {
    "angry": 0, "fear": 1, "happy": 2,
    "neutral": 3, "sad": 4, "surprise": 5
}

# ===================== PREPROCESS =====================
def preprocess(landmarks):
    landmarks = torch.tensor(landmarks, dtype=torch.float32)
    center = landmarks[1]
    centered = landmarks - center
    left_eye = centered[33]
    right_eye = centered[263]
    scale = torch.norm(left_eye - right_eye) + 1e-6
    centered = centered / scale

    features = []
    for p in centered:
        x, y, z = p
        dist = torch.sqrt(x**2 + y**2)
        angle = torch.atan2(y, x)
        features.append([x.item(), y.item(), z.item(), dist.item(), angle.item()])

    x = torch.tensor(features)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x

# ===================== EDGE INDEX =====================
def get_edge_index():
    edges = []
    def add_edges(pairs):
        for a, b in pairs:
            edges.append([a, b])
            edges.append([b, a])

    face_oval = [10,338,297,332,284,251,389,356,454,323,361,288,
                 397,365,379,378,400,377,152,148,176,149,150,136,
                 172,58,132,93,234,127,162,21,54,103,67,109]
    add_edges(list(zip(face_oval, face_oval[1:])))
    add_edges([(face_oval[-1], face_oval[0])])

    lips_outer = [61,146,91,181,84,17,314,405,321,375,291,308]
    add_edges(list(zip(lips_outer, lips_outer[1:])))
    add_edges([(lips_outer[-1], lips_outer[0])])

    return torch.tensor(edges, dtype=torch.long).t().contiguous()

base_edge_index = get_edge_index()

# ===================== DATASET =====================
class FacialDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, is_train=False):
        self.samples = []
        self.is_train = is_train  # 🔥 track split

        for label_name in os.listdir(root_dir):
            print("reading:", label_name)
            if label_name not in label_map:
                continue
            label_path = os.path.join(root_dir, label_name)
            for file in os.listdir(label_path):
                try:
                    with open(os.path.join(label_path, file), "r") as f:
                        data = json.load(f)
                        x = preprocess(data["landmarks"])
                        y = label_map[label_name]
                        # 🔥 Pre-compute KNN edges once per sample
                        knn_edges = knn_graph(x, k=8)
                        edge_index = torch.cat([base_edge_index, knn_edges], dim=1)
                        edge_index, _ = add_self_loops(edge_index)
                        self.samples.append((x, y, edge_index))
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, edge_index = self.samples[idx]

        # 🔥 Augmentation only during training, fresh each epoch
        if self.is_train:
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.01   # jitter
            if random.random() < 0.3:
                x = x + torch.randn_like(x) * 0.005  # subtler second pass

        return Data(x=x, edge_index=edge_index, y=torch.tensor(y))

# ===================== FOCAL LOSS WITH CLASS WEIGHTS =====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # 🔥 class weights

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

# ===================== MODEL =====================
class GNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = GATConv(5, 64, heads=4, dropout=0.3)
        self.bn1 = BatchNorm(256)

        self.conv2 = GATConv(256, 128, heads=2, dropout=0.3)
        self.bn2 = BatchNorm(256)

        self.conv3 = GATConv(256, 128, heads=1, dropout=0.3)
        self.bn3 = BatchNorm(128)  # 🔥 added

        self.fc1 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)   # 🔥 extra FC layer
        self.bn5 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)   # 🔥 now normalized
        x = F.elu(x)

        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x1, x2], dim=1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)   # 🔥
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.fc3(x)

# ===================== TRAIN =====================
def train():
    dataset_path = "../my_react_app/public/train3_landmarks"
    print("loading dataset")
    full_dataset = FacialDataSet(dataset_path, is_train=False)  # load all first
    print("dataset loaded")

    labels = [y for _, y, _ in full_dataset.samples]
    print("Class distribution:", Counter(labels))

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    # 🔥 Separate datasets with correct is_train flag
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)

    # Patch is_train for the underlying dataset per subset
    # Simpler: just override __getitem__ via a wrapper
    class SplitDataset(torch.utils.data.Dataset):
        def __init__(self, subset, is_train):
            self.subset = subset
            self.is_train = is_train
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            data = self.subset[idx]
            if self.is_train:
                x = data.x
                if random.random() < 0.5:
                    x = x + torch.randn_like(x) * 0.01
                if random.random() < 0.3:
                    x = x + torch.randn_like(x) * 0.005
                data = Data(x=x, edge_index=data.edge_index, y=data.y)
            return data

    train_loader = DataLoader(
        SplitDataset(train_dataset, is_train=True),
        batch_size=32, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        SplitDataset(val_dataset, is_train=False),
        batch_size=32, num_workers=2
    )

    # 🔥 Compute class weights to counter imbalance
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(label_counts) * label_counts[i]) for i in range(len(label_map))],
        dtype=torch.float32
    ).to(device)
    print("Class weights:", class_weights)

    model = GNN(len(label_map)).to(device)

    criterion = FocalLoss(gamma=2, weight=class_weights)  # 🔥 weighted focal loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 🔥 Cosine annealing — doesn't kill LR prematurely
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    best_val = 0
    patience = 12  # 🔥 slightly more generous
    wait = 0

    for epoch in range(60):
        model.train()
        train_correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

        train_acc = train_correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total
        scheduler.step()  # cosine just steps each epoch

        print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping | Best val: {best_val:.4f}")
                break

    torch.save(model.state_dict(), "expression.pth")
    print("Done!")

if __name__ == "__main__":
    train()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader,Dataset,random_split
# import os
# import json
# import math
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# label_map = {
#     "angry":0,
#     "fear":1,
#     "happy":2,
#     "neutral":3,
#     "sad":4,
#     "surprise":5
# }
# def preprocess(landmarks):
#     # -------- CENTER --------
#     ref = landmarks[0]
#     centered = [[x - ref[0], y - ref[1]] for x, y, z in landmarks]

#     # -------- SCALE --------
#     left_eye = centered[33]
#     right_eye = centered[263]

#     scale = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) + 1e-8

#     scaled = [[x/scale, y/scale] for x, y in centered]

#     # -------- FLATTEN --------
#     flat = []
#     for point in scaled:
#         flat.extend(point)

#     x = torch.tensor(flat, dtype=torch.float32)

#     # -------- OPTIONAL NORMALIZATION --------
#     x = (x - x.mean()) / (x.std() + 1e-8)

#     return x

# class FacialDataSet(Dataset):
#     def __init__(self,root_dir):
#         self.samples=[]
#         for label_name in os.listdir(root_dir):
#             label_path=os.path.join(root_dir,label_name)

#             if label_name not in label_map:
#                 continue

#             for file in os.listdir(label_path):
#                 file_path = os.path.join(label_path,file)

#                 try:
#                     with open(file_path,"r") as f:
#                         data=json.load(f)

#                         features= preprocess(data["landmarks"])
#                         labels=label_map[label_name]

#                         self.samples.append((features,labels))
#                 except:
#                     continue

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self,index):
#         x,y=self.samples[index]

#         # x=torch.tensor(x,dtype=torch.float32)
#         y=torch.tensor(y,dtype=torch.long)
#         return x,y

# class MLP(nn.Module):
#     def __init__(self,input_size,num_classes):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Linear(input_size,256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(256,128),
#             nn.ReLU(),

#             nn.Linear(128,num_classes)
#         )

#     def forward(self,x):
#         return self.model(x)

# def train():
#     dataset_path = "../my_react_app/public/train3_landmarks"

#     dataset = FacialDataSet(dataset_path)
#     # dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

#     input_size = len(dataset[0][0])
#     num_classes=len(label_map)

#     model=MLP(input_size,num_classes).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-3)

#     train_size = int(0.8*len(dataset))
#     val_size=len(dataset) - train_size

#     train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
#     train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2,pin_memory=True)
#     val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size=32,num_workers=2,pin_memory=True)

#     epochs=50
#     best_val_acc = 0
#     for epoch in range(epochs):
#         model.train()

#         train_loss = 0
#         train_correct = 0
#         train_total = 0

#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
#             outputs = model(x)
#             loss = criterion(outputs, y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()* x.size(0)

#             _, pred = torch.max(outputs, 1)
#             train_correct += (pred == y).sum().item()
#             train_total += y.size(0)
#         train_loss = train_loss / train_total
#         train_acc = train_correct / train_total

#         # ================= VALIDATION =================
#         model.eval()

#         val_loss = 0
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for x, y in val_dataloader:
#                 x = x.to(device)
#                 y = y.to(device)
#                 outputs = model(x)
#                 loss = criterion(outputs, y)

#                 val_loss += loss.item()*x.size(0)

#                 _, pred = torch.max(outputs, 1)
#                 val_correct += (pred == y).sum().item()
#                 val_total += y.size(0)
#         val_loss=val_loss/val_total
#         val_acc = val_correct / val_total
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pth")
#         print(f"""Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f} """)

#     torch.save(model.state_dict(), "expression.pth")
#     print("model saved")

# if __name__=="__main__":
#     train()
