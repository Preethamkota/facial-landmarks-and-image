import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import os
import json
import math
label_map = {
    "anger":0,
    "fear":1,
    "joy":2,
    "Natural":3,
    "sadness":4,
    "surprise":5
}
def preprocess(landmarks):
    # -------- CENTER --------
    ref = landmarks[0]  # nose
    centered = [[x - ref[0], y - ref[1]] for x, y, z in landmarks]  # remove Z

    # -------- SCALE --------
    left_eye = centered[33]
    right_eye = centered[263]

    scale = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) + 1e-8

    scaled = [[x/scale, y/scale] for x, y in centered]

    # -------- FLATTEN --------
    flat = []
    for point in scaled:
        flat.extend(point)   # only x,y

    x = torch.tensor(flat, dtype=torch.float32)

    # -------- OPTIONAL NORMALIZATION --------
    x = (x - x.mean()) / (x.std() + 1e-8)

    return x

class FacialDataSet():
    def __init__(self,root_dir):
        self.samples=[]
        for label_name in os.listdir(root_dir):
            label_path=os.path.join(root_dir,label_name)

            if label_name not in label_map:
                continue

            for file in os.listdir(label_path):
                file_path = os.path.join(label_path,file)

                try:
                    with open(file_path,"r") as f:
                        data=json.load(f)

                        features= preprocess(data["landmarks"])
                        labels=label_map[label_name]

                        self.samples.append((features,labels))
                except:
                    continue

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        x,y=self.samples[index]

        x=torch.tensor(x,dtype=torch.float32)
        y=torch.tensor(y,dtype=torch.long)
        return x,y

class MLP(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.model(x)

def train():
    dataset_path = "../my-react-app/public/train_landmarks"

    dataset = FacialDataSet(dataset_path)
    # dataloader = DataLoader(dataset,batch_size=32,shuffle=True)

    input_size = len(dataset[0][0])
    num_classes=len(label_map)

    model=MLP(input_size,num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-4)

    train_size = int(0.8*len(dataset))
    val_size=len(dataset) - train_size

    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_dataloader = DataLoader(val_dataset,shuffle=False,batch_size=32)

    epochs=75

    for epoch in range(epochs):
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total

        # ================= VALIDATION =================
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        best_val_acc = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
        print(f"""Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f} """)

    torch.save(model.state_dict(), "expression.pth")
    print("model saved")

if __name__=="__main__":
    train()