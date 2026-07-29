"""
Step 2.3 + 2.4: Train EEGNet on BCI IV-2a using the native session-based
train/test split. Refactored into a reusable train_subject() function
so it can be called for a single subject (CLI) or looped across all
9 subjects (train_all_subjects.py).
"""
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from eegnet import EEGNet  # same folder import


class EEGDataset(Dataset):
    def __init__(self, X, y):
        # X: (n_trials, n_channels, n_timepoints) -> add channel dim for Conv2d
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, C, T)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split_data(subject_id: int = 1):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=4, fmax=38, resample=250)

    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    # Encode string labels -> integers, consistently
    classes = sorted(set(y))  # ['feet', 'left_hand', 'right_hand', 'tongue']
    label_map = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([label_map[label] for label in y])

    train_mask = metadata["session"] == "0train"
    test_mask = metadata["session"] == "1test"

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]

    return (X_train, y_train), (X_test, y_test), label_map


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def train_subject(subject_id: int, n_epochs: int = 100, verbose: bool = True):
    """Train EEGNet for one subject and return results dict. Reusable by both
    single-subject CLI use and the multi-subject driver script."""
    device = get_device()
    if verbose:
        print(f"\n{'='*60}\nSubject {subject_id} | Using device: {device}\n{'='*60}")

    (X_train, y_train), (X_test, y_test), label_map = load_split_data(subject_id=subject_id)
    if verbose:
        print(f"Train trials: {len(y_train)} | Test trials: {len(y_test)}")
        print(f"Label mapping: {label_map}")

    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = EEGNet(n_classes=4, n_channels=X_train.shape[1], n_timepoints=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    checkpoint_path = f"../../models/eegnet_subject{subject_id}.pt"
    os.makedirs("../../models", exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

        train_loss /= total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "n_channels": X_train.shape[1],
                "n_timepoints": X_train.shape[2],
                "val_acc": val_acc,
                "subject_id": subject_id,
            }, checkpoint_path)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if verbose:
        print(f"Subject {subject_id} best val_acc: {best_val_acc:.3f}")
        print(f"Checkpoint saved to: {checkpoint_path}")

    # Log to results CSV (appends one row per subject run)
    log_path = "../../logs/training_results.csv"
    os.makedirs("../../logs", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["subject_id", "n_train_trials", "n_test_trials", "best_val_acc", "timestamp"])
        writer.writerow([subject_id, len(y_train), len(y_test), round(best_val_acc, 4),
                          datetime.now().isoformat(timespec="seconds")])

    return {
        "subject_id": subject_id,
        "best_val_acc": best_val_acc,
        "checkpoint_path": checkpoint_path,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train_subject(subject_id=args.subject, n_epochs=args.epochs)
