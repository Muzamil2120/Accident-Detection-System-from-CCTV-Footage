from __future__ import annotations

"""Beginner-friendly CNN + LSTM training pipeline for accident / no-accident classification.

What this does
- Loads frame-sequences created by tools/extract_sequences.py
- Trains a small CNN (spatial features) + LSTM (temporal features)
- Evaluates on a test split with precision/recall + confusion matrix
- Saves the trained model into models/trained/

Dataset folder structure (created by extract_sequences.py)
Datasets/
  raw/
    accident/
    no_accident/
  sequences/
    accident/<video>__seq000/000.jpg ...
    no_accident/<video>__seq000/000.jpg ...
  index.csv

Why this is reliable-ish (but not perfect)
- Temporal modeling helps reduce false positives from slow traffic.
- We explicitly evaluate precision/recall so you can see false alarms vs misses.

Requirements
- PyTorch (torch, torchvision)
- scikit-learn
- opencv-python

Install training deps (example):
  pip install -r requirements-train.txt
  # then install torch/torchvision from https://pytorch.org/get-started/locally/
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch is not installed. Install torch/torchvision using the official selector: "
            "https://pytorch.org/get-started/locally/"
        ) from exc


_require_torch()

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {"no_accident": 0, "accident": 1}
ID_TO_LABEL = {0: "no_accident", 1: "accident"}


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 112
    seq_len: int = 16
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 0

    # split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1


class SequenceDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], image_size: int, seq_len: int) -> None:
        self.items = items
        self.image_size = image_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq_dir, label = self.items[idx]
        frame_paths = sorted(seq_dir.glob("*.jpg"))
        if len(frame_paths) < self.seq_len:
            raise RuntimeError(f"Sequence {seq_dir} has {len(frame_paths)} frames, expected >= {self.seq_len}")

        # Load the first `seq_len` frames so the CNN+LSTM stack sees a consistent temporal window.
        frames = []
        for fp in frame_paths[: self.seq_len]:
            bgr = cv2.imread(str(fp))
            if bgr is None:
                raise RuntimeError(f"Failed to read frame: {fp}")
            bgr = cv2.resize(bgr, (self.image_size, self.image_size))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)

        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,C)
        # Convert to torch (T,C,H,W)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        return tensor, torch.tensor(label, dtype=torch.long)


class FrameCNN(nn.Module):
    """Small CNN encoder (NOT advanced), outputs an embedding per frame."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W)
        f = self.features(x).flatten(1)  # (B,64)
        return self.proj(f)  # (B,E)


class CnnLstmClassifier(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.cnn = FrameCNN(embed_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W)
        b, t, c, h, w = x.shape
        x2 = x.reshape(b * t, c, h, w)
        emb = self.cnn(x2)  # (B*T,E)
        emb = emb.reshape(b, t, -1)  # (B,T,E)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]  # last timestep
        logits = self.head(last)
        return logits


def load_index(index_csv: Path) -> list[tuple[Path, int]]:
    if not index_csv.exists():
        raise FileNotFoundError(
            f"{index_csv} not found. First run tools/extract_sequences.py to create sequences and index.csv"
        )

    items: list[tuple[Path, int]] = []
    with index_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_dir = Path(row["sequence_dir"])
            label_str = row["label"].strip()
            if label_str not in LABEL_TO_ID:
                continue
            items.append((seq_dir, LABEL_TO_ID[label_str]))

    return items


def split_items(items: list[tuple[Path, int]], cfg: TrainConfig) -> tuple[list, list, list]:
    # Shuffle once so the label ratios stay roughly balanced across splits.
    random.seed(cfg.seed)
    items = items.copy()
    random.shuffle(items)

    n = len(items)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "Datasets"),
        help="Path to Datasets/ folder",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=112)

    args = parser.parse_args()

    cfg = TrainConfig(
        image_size=args.image_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    datasets_dir = Path(args.datasets_dir)
    index_csv = datasets_dir / "index.csv"
    items = load_index(index_csv)

    if not items:
        raise RuntimeError("No sequences found in index.csv. Add videos to Datasets/raw/* and run extract_sequences.py")

    train_items, val_items, test_items = split_items(items, cfg)

    train_ds = SequenceDataset(train_items, image_size=cfg.image_size, seq_len=cfg.seq_len)
    val_ds = SequenceDataset(val_items, image_size=cfg.image_size, seq_len=cfg.seq_len)
    test_ds = SequenceDataset(test_items, image_size=cfg.image_size, seq_len=cfg.seq_len)

    # Wrap each split in a DataLoader so PyTorch batches and shuffles the tensors.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CnnLstmClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_recall = -1.0
    best_path = Path(__file__).resolve().parent / "models" / "trained" / "cnn_lstm_accident_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Run full epochs with forward/backward passes and validation checks.
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item())

        # Evaluate the current weights to track precision and recall on the validation split.
        val_metrics = evaluate(model, val_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | loss={avg_loss:.4f} | "
            f"val_precision={val_metrics['precision']:.3f} | val_recall={val_metrics['recall']:.3f}"
        )

        # We prioritize recall a bit (missing accidents is usually worse than a false alarm),
        # but you can change this to precision or F1 depending on your product requirements.
        if val_metrics["recall"] > best_val_recall:
            best_val_recall = val_metrics["recall"]
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, best_path)

    # Final evaluation on test split
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, device)

    out_dir = Path(__file__).resolve().parent / "models" / "trained"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "cnn_lstm_metrics.json"
    metrics = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": str(device),
        "val_best_recall": float(best_val_recall),
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels": ID_TO_LABEL,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save confusion matrix as a simple CSV too (easy to open in Excel)
    cm = np.array(test_metrics["confusion_matrix"], dtype=int)
    cm_csv = out_dir / "cnn_lstm_confusion_matrix.csv"
    cm_csv.write_text(
        "pred_no_accident,pred_accident\n" + f"{cm[0,0]},{cm[0,1]}\n{cm[1,0]},{cm[1,1]}\n",
        encoding="utf-8",
    )

    print("\nTraining complete.")
    print(f"Saved best model: {best_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix CSV: {cm_csv}")


if __name__ == "__main__":
    main()
