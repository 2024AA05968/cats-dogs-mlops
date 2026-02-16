from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------
# Windows: stabilize DLL loading for torch (fixes WinError 1114 / c10.dll)
# ---------------------------
if sys.platform.startswith("win"):
    try:
        # Add torch's DLL folder explicitly before importing torch
        # This avoids DLL init failures caused by path resolution / dependency order.
        import site

        sp = site.getsitepackages()[0]
        torch_lib = Path(sp) / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        # If this fails, torch may still import fine; we don't hard-fail here.
        pass

# Import torch BEFORE mlflow (mlflow will be imported lazily later)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
except ImportError:
    yaml = None

from .utils import set_seed


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


def get_dataloaders(cfg: Config):
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


def setup_mlflow(cfg: Config) -> None:
    # Lazy import to avoid Windows DLL-load interference
    import mlflow

    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def main():
    cfg = load_config()

    # NOTE: On Windows, DataLoader multiprocessing can be flaky.
    # If you hit issues later, set num_workers: 0 in configs/train.yaml
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = get_dataloaders(cfg)

    model = BaselineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    setup_mlflow(cfg)

    # Lazy import here as well (ensures torch loads first)
    import mlflow

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
        })
        mlflow.log_dict(class_to_idx, "class_to_idx.json")

        print("[INFO] Setup complete. Training loop will be added in Step 9.2.2.")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"[INFO] Classes: {class_to_idx}")


if __name__ == "__main__":
    main()