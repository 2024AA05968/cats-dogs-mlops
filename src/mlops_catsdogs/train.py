from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------
# Windows: stabilize DLL loading for torch (fixes WinError 1114 / c10.dll)
# ---------------------------
if sys.platform.startswith("win"):
    try:
        import site

        sp = site.getsitepackages()[0]
        torch_lib = Path(sp) / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        # Not fatal; torch may still load fine
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

try:
    import yaml
except ImportError:
    yaml = None

from .utils import set_seed


# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    seed: int
    data_root: str
    img_size: int
    num_workers: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    experiment_name: str
    run_name: str
    tracking_uri: str


def load_config(path: str = "configs/train.yaml") -> Config:
    if yaml is None:
        raise RuntimeError("PyYAML missing. Install with: pip install pyyaml==6.0.2")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        seed=int(raw["seed"]),
        data_root=str(raw["data"]["root"]),
        img_size=int(raw["data"]["img_size"]),
        num_workers=int(raw["data"]["num_workers"]),
        batch_size=int(raw["train"]["batch_size"]),
        epochs=int(raw["train"]["epochs"]),
        lr=float(raw["train"]["lr"]),
        weight_decay=float(raw["train"]["weight_decay"]),
        experiment_name=str(raw["mlflow"]["experiment_name"]),
        run_name=str(raw["mlflow"]["run_name"]),
        tracking_uri=str(raw["mlflow"].get("tracking_uri") or ""),
    )


# ---------------------------
# Model
# ---------------------------
class BaselineCNN(nn.Module):
    """A simple CNN baseline for 224x224 RGB images."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------
# Data
# ---------------------------
def get_dataloaders(cfg: Config):
    # Augmentation only for training set
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
    ])

    train_dir = Path(cfg.data_root) / "train"
    val_dir = Path(cfg.data_root) / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found. Expected:\n  {train_dir}\n  {val_dir}\n"
            "Ensure Step 8 preprocessing completed successfully."
        )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, train_ds.class_to_idx


# ---------------------------
# MLflow (lazy import)
# ---------------------------
def setup_mlflow(cfg: Config) -> None:
    import mlflow  # lazy import to avoid DLL conflicts on Windows

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = loss_sum / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = loss_sum / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc, y_true, y_pred


# ---------------------------
# Plotting artifacts
# ---------------------------
def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_path = out_dir / "loss_curve.png"
    acc_path = out_dir / "acc_curve.png"

    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="train_acc")
    plt.plot(val_accs, label="val_acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    return loss_path, acc_path


def plot_confusion(y_true, y_pred, class_to_idx: Dict[str, int], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / "confusion_matrix.png"

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Val)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    return cm_path


# ---------------------------
# Main
# ---------------------------
def main():
    cfg = load_config()

    # NOTE: On Windows, keep num_workers=0 for stability (you set this in train.yaml)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = get_dataloaders(cfg)

    model = BaselineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    setup_mlflow(cfg)
    import mlflow  # lazy import after torch is already loaded safely

    artifacts_dir = Path("artifacts") / "baseline_cnn"
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "seed": cfg.seed,
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "model": "BaselineCNN",
            "device": str(device),
            "num_workers": cfg.num_workers,
        })
        mlflow.log_dict(class_to_idx, "class_to_idx.json")

        print(f"[INFO] Training on device: {device}")
        for epoch in range(1, cfg.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, y_true, y_pred = eval_one_epoch(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(va_loss)
            train_accs.append(tr_acc)
            val_accs.append(va_acc)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            }, step=epoch)

            print(
                f"[EPOCH {epoch}/{cfg.epochs}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
            )

        # Artifacts: curves + confusion matrix + classification report
        loss_path, acc_path = plot_curves(train_losses, val_losses, train_accs, val_accs, artifacts_dir)
        cm_path = plot_confusion(y_true, y_pred, class_to_idx, artifacts_dir)

        report_text = classification_report(y_true, y_pred, target_names=["cat", "dog"])
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        mlflow.log_artifact(str(loss_path))
        mlflow.log_artifact(str(acc_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))

        # Save model artifact
        model_path = models_dir / "baseline_cnn.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "img_size": cfg.img_size,
        }, model_path)

        mlflow.log_artifact(str(model_path))

        print(f"[SUCCESS] Saved model to: {model_path}")
        print("[SUCCESS] Logged metrics and artifacts to MLflow.")


if __name__ == "__main__":
    main()