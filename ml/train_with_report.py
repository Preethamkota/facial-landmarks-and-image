from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import FacialDataSet, MLP, label_map
from training_report import create_training_report


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "my-react-app" / "public" / "train_landmarks"
REPORTS_DIR = ROOT_DIR / "my-react-app" / "public"
BEST_MODEL_PATH = ROOT_DIR / "ml" / "best_model.pth"
FINAL_MODEL_PATH = ROOT_DIR / "ml" / "expression.pth"


def build_confusion_matrix(targets, predictions, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, predicted_label in zip(targets, predictions):
        matrix[true_label][predicted_label] += 1
    return matrix


def train_with_report():
    dataset = FacialDataSet(str(DATASET_PATH))
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {DATASET_PATH}")

    input_size = len(dataset[0][0])
    num_classes = len(label_map)

    model = MLP(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    epochs = 50
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    final_targets = []
    final_predictions = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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

        train_loss = train_loss / max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_targets = []
        epoch_predictions = []

        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                epoch_targets.extend(y.tolist())
                epoch_predictions.extend(pred.tolist())

        val_loss = val_loss / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        final_targets = epoch_targets
        final_predictions = epoch_predictions

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    labels = list(label_map.keys())
    confusion_matrix = build_confusion_matrix(final_targets, final_predictions, len(labels))
    report_path = create_training_report(
        output_dir=REPORTS_DIR,
        labels=labels,
        history=history,
        confusion_matrix=confusion_matrix,
        summary={
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc": history["train_acc"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_acc": history["val_acc"][-1],
            "best_val_acc": best_val_acc,
            "best_model_path": str(BEST_MODEL_PATH),
            "final_model_path": str(FINAL_MODEL_PATH),
        },
    )

    print(f"model saved to {FINAL_MODEL_PATH}")
    print(f"training report saved to {report_path}")


if __name__ == "__main__":
    train_with_report()
