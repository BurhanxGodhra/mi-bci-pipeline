"""
Step 8-fix: Corrected training methodology.

Three-way split, strictly enforced:
  - Inner-train (from session '0train'): used for gradient updates
  - Inner-validation (from session '0train'): used for checkpoint selection
  - True test (session '1test'): touched EXACTLY ONCE, after training is
    fully complete, to compute the number that gets reported

Previously, checkpoint selection and final reporting both used session
'1test' -- this is test-set leakage through model selection. This file
corrects that. See docs/decisions.md and docs/EXTERNAL_REVIEW.md.
"""
import csv
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from eegnet import EEGNet


class EEGDataset(Dataset):
    def __init__(self, X, y):
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

    classes = sorted(set(y))
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
    """Train EEGNet for one subject with a proper three-way split.

    Checkpoint selection uses ONLY the inner validation split (carved from
    session '0train'). The true test set (session '1test') is evaluated
    exactly once, after training completes, and that number -- final_test_acc
    -- is what gets reported and logged.
    """
    device = get_device()
    if verbose:
        print(f"\n{'='*60}\nSubject {subject_id} | Using device: {device}\n{'='*60}")

    (X_train_full, y_train_full), (X_test, y_test), label_map = load_split_data(subject_id=subject_id)

    # Three-way split. X_test is NOT touched again until final evaluation below.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )

    if verbose:
        print(f"Inner-train: {len(y_train)} | Inner-val: {len(y_val)} | Held-out test: {len(y_test)}")
        print(f"Label mapping: {label_map}")

    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=32, shuffle=False)

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
        # Checkpoint selection uses ONLY the inner validation split -- never the test set.
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "n_channels": X_train.shape[1],
                "n_timepoints": X_train.shape[2],
                "internal_val_acc": val_acc,
                "subject_id": subject_id,
            }, checkpoint_path)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | train_acc={train_acc:.3f} "
                  f"| inner_val_acc={val_acc:.3f} (best={best_val_acc:.3f})")

    # --- Final evaluation: session '1test' touched EXACTLY ONCE, here ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=32, shuffle=False)
    test_loss, final_test_acc = evaluate(model, test_loader, device, criterion)

    # Update the checkpoint with the true, held-out test accuracy
    checkpoint["final_test_acc"] = final_test_acc
    torch.save(checkpoint, checkpoint_path)

    if verbose:
        print(f"Subject {subject_id} | best internal val_acc: {best_val_acc:.3f} "
              f"| FINAL HELD-OUT TEST ACC: {final_test_acc:.3f}")
        print(f"Checkpoint saved to: {checkpoint_path}")

    log_path = "../../logs/training_results.csv"
    os.makedirs("../../logs", exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["subject_id", "n_train_inner", "n_val_inner", "n_test",
                              "best_internal_val_acc", "final_test_acc", "timestamp"])
        writer.writerow([subject_id, len(y_train), len(y_val), len(y_test),
                          round(best_val_acc, 4), round(final_test_acc, 4),
                          datetime.now().isoformat(timespec="seconds")])

    return {
        "subject_id": subject_id,
        "best_internal_val_acc": best_val_acc,
        "final_test_acc": final_test_acc,
        "checkpoint_path": checkpoint_path,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train_subject(subject_id=args.subject, n_epochs=args.epochs)

def train_subject_refit(subject_id: int, n_epochs: int = 100, verbose: bool = True):
    """Two-stage training: (1) determine the best stopping epoch via a proper
    train/val split, discarding that model; (2) retrain from scratch on the
    FULL training session using that stopping point, evaluated once against
    the true held-out test set. Recovers the training-data disadvantage from
    the internal validation split without reintroducing any test-set leakage."""
    device = get_device()
    (X_train_full, y_train_full), (X_test, y_test), label_map = load_split_data(subject_id=subject_id)

    # --- Stage 1: determine best stopping epoch via honest validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    train_loader = DataLoader(EEGDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_val, y_val), batch_size=32, shuffle=False)

    probe_model = EEGNet(n_classes=4, n_channels=X_train.shape[1], n_timepoints=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_epoch = 0.0, 1
    for epoch in range(1, n_epochs + 1):
        probe_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(probe_model(xb), yb)
            loss.backward()
            optimizer.step()
        _, val_acc = evaluate(probe_model, val_loader, device, criterion)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch

    if verbose:
        print(f"Subject {subject_id} | Stage 1: best stopping point = epoch {best_epoch} "
              f"(val_acc={best_val_acc:.3f})")

    # --- Stage 2: refit from scratch on FULL training session, stop at best_epoch ---
    full_loader = DataLoader(EEGDataset(X_train_full, y_train_full), batch_size=32, shuffle=True)
    final_model = EEGNet(n_classes=4, n_channels=X_train.shape[1], n_timepoints=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, best_epoch + 1):
        final_model.train()
        for xb, yb in full_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            optimizer.step()

    checkpoint_path = f"../../models/eegnet_subject{subject_id}.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(), "label_map": label_map,
        "n_channels": X_train.shape[1], "n_timepoints": X_train.shape[2],
        "stopping_epoch_from_validation": best_epoch, "subject_id": subject_id,
    }, checkpoint_path)

    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=32, shuffle=False)
    _, final_test_acc = evaluate(final_model, test_loader, device, criterion)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint["final_test_acc"] = final_test_acc
    torch.save(checkpoint, checkpoint_path)

    if verbose:
        print(f"Subject {subject_id} | Stage 2 refit (epochs=1..{best_epoch}, n=288) "
              f"| FINAL HELD-OUT TEST ACC: {final_test_acc:.3f}")

    log_path = "../../logs/training_results.csv"
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["subject_id", "method", "stopping_epoch", "final_test_acc", "timestamp"])
        writer.writerow([subject_id, "refit_full_data", best_epoch, round(final_test_acc, 4),
                          datetime.now().isoformat(timespec="seconds")])

    return {"subject_id": subject_id, "final_test_acc": final_test_acc, "stopping_epoch": best_epoch}