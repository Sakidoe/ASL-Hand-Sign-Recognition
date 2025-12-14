# Train baseline MLP and a simple CNN on Sign Language MNIST.
# run this file to train then run the test file 
# running the webcame demo is optional to test live movement
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# mapping for the Kaggle labels (J and Z not in dataset)
LABEL_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R",
    18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
}


def index_to_letter(idx: int) -> str:
    # return '?' for labels not used (9 and 25)
    return LABEL_TO_LETTER.get(idx, "?")


# ================ data helpers ================
def load_sign_mnist(csv_path: str) -> TensorDataset:
    # expects 'label' column then 784 pixel columns
    df = pd.read_csv(csv_path)
    labels = df["label"].values.astype(np.int64)
    pixels = df.drop(columns=["label"]).values.astype(np.float32)
    pixels /= 255.0
    images = pixels.reshape(-1, 1, 28, 28)
    x_tensor = torch.tensor(images, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def train_val_split(dataset: TensorDataset, val_ratio: float = 0.2
                    ) -> Tuple[TensorDataset, TensorDataset]:
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    return random_split(dataset, [train_size, val_size])


# ================ models ================
class BaselineMLP(nn.Module):
    # simple fully connected baseline
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    # small CNN for 28x28 grayscale images
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ================ helpers ================
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).sum().item() / labels.size(0)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)
        batches += 1
    return running_loss / batches, running_acc / batches


def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            running_acc += accuracy_from_logits(logits, labels)
            batches += 1
    return running_loss / batches, running_acc / batches


def train_model(model, train_loader, val_loader, epochs: int, lr: float,
                save_path: str):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}  Train loss {train_loss:.4f} acc {train_acc:.4f}  Val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  saved {save_path} (best val acc {best_val_acc:.4f})")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


# ================ main ================
def main():
    csv_path = "sign_mnist_train.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in this folder")

    dataset = load_sign_mnist(csv_path)
    train_ds, val_ds = train_val_split(dataset, val_ratio=0.2)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    print("\nTraining baseline MLP...")
    baseline = BaselineMLP(num_classes=26)
    train_model(baseline, train_loader, val_loader, epochs=5, lr=1e-3, save_path="asl_baseline.pth")

    print("\nTraining CNN...")
    cnn = SimpleCNN(num_classes=26)
    train_model(cnn, train_loader, val_loader, epochs=10, lr=1e-3, save_path="asl_cnn.pth")


if __name__ == "__main__":
    main()
