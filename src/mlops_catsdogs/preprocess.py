# src/mlops_catsdogs/preprocess.py
import os
import sys
import shutil
import random
from pathlib import Path
from typing import Tuple, List

from PIL import Image, UnidentifiedImageError

# -------- Configuration (you can keep these defaults or override via CLI) --------
RAW_DIR = Path("data/raw/PetImages")
OUT_DIR = Path("data/processed")
IMG_SIZE: Tuple[int, int] = (224, 224)
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
LABEL_MAP = {"Cat": "cat", "Dog": "dog"}  # normalize folder names
SEED = 42

# ---------------------------------------------------------------------------------


def is_image_ok(path: Path) -> bool:
    """Return True if the image can be opened and converted; False if corrupted."""
    try:
        with Image.open(path) as img:
            img.verify()  # quick check
        # Reopen for actual conversion (verify() leaves fp at EOF)
        with Image.open(path) as img:
            _ = img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def find_images() -> List[Tuple[Path, str]]:
    """
    Scan RAW_DIR/Cat and RAW_DIR/Dog and collect valid image paths with labels.
    Returns a list of tuples (path, label_str) where label_str in {"cat", "dog"}.
    """
    samples: List[Tuple[Path, str]] = []
    for raw_label in ["Cat", "Dog"]:
        label_str = LABEL_MAP[raw_label]
        label_dir = RAW_DIR / raw_label
        if not label_dir.exists():
            print(f"[WARN] Missing directory: {label_dir} — skipping.", file=sys.stderr)
            continue
        for p in label_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if is_image_ok(p):
                    samples.append((p, label_str))
                else:
                    print(f"[WARN] Corrupted image skipped: {p}", file=sys.stderr)
    return samples


def stratified_split(
    samples: List[Tuple[Path, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
):
    """
    Simple stratified split without sklearn.
    Returns dict: split_name -> list[(Path, label)]
    """
    random.seed(seed)
    by_class = {}
    for p, lbl in samples:
        by_class.setdefault(lbl, []).append(p)

    splits = {"train": [], "val": [], "test": []}
    for lbl, paths in by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_paths = paths[:n_train]
        val_paths = paths[n_train : n_train + n_val]
        test_paths = paths[n_train + n_val :]

        splits["train"].extend([(p, lbl) for p in train_paths])
        splits["val"].extend([(p, lbl) for p in val_paths])
        splits["test"].extend([(p, lbl) for p in test_paths])

        print(
            f"[INFO] Class '{lbl}': total={n}, train={len(train_paths)}, "
            f"val={len(val_paths)}, test={len(test_paths)}"
        )
    return splits


def prepare_out_dirs():
    """Create clean output directory structure."""
    if OUT_DIR.exists():
        print(f"[INFO] Removing existing directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val", "test"]:
        for lbl in ["cat", "dog"]:
            out_path = OUT_DIR / split / lbl
            out_path.mkdir(parents=True, exist_ok=True)


def process_and_copy(src_path: Path, dst_path: Path):
    """Open, convert to RGB, resize to IMG_SIZE, and save to dst_path."""
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        img.save(dst_path, format="JPEG", quality=95)


def main():
    # Validate ratios
    if abs(sum(SPLITS.values()) - 1.0) > 1e-6:
        print("[ERROR] SPLITS must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Scanning images under: {RAW_DIR}")
    samples = find_images()
    if not samples:
        print("[ERROR] No valid images found. Check your RAW_DIR.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(samples)} valid images. Creating stratified splits...")
    split_data = stratified_split(samples, SPLITS["train"], SPLITS["val"], seed=SEED)

    print(f"[INFO] Preparing output directories under: {OUT_DIR}")
    prepare_out_dirs()

    print("[INFO] Processing and copying images...")
    counter = 0
    for split_name, items in split_data.items():
        for src_path, lbl in items:
            # Build deterministic destination filename
            # e.g., train/cat/cat_000001.jpg
            counter += 1
            fname = f"{lbl}_{counter:06d}.jpg"
            dst = OUT_DIR / split_name / lbl / fname
            try:
                process_and_copy(src_path, dst)
            except Exception as e:
                print(f"[WARN] Failed to process {src_path}: {e}", file=sys.stderr)

    # Summary
    for split_name in ["train", "val", "test"]:
        cat_n = len(list((OUT_DIR / split_name / "cat").glob("*.jpg")))
        dog_n = len(list((OUT_DIR / split_name / "dog").glob("*.jpg")))
        print(
            f"[DONE] {split_name}: cat={cat_n}, dog={dog_n}, "
            f"total={cat_n + dog_n} (images resized to {IMG_SIZE})"
        )

    print("[SUCCESS] Preprocessing completed.")


if __name__ == "__main__":
    main()