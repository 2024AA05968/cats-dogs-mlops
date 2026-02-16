from pathlib import Path
import hashlib
import random
import shutil
import sys
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError

RAW_DIR = Path("data/raw/PetImages")
OUT_DIR = Path("data/processed")
IMG_SIZE = (224, 224)

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
LABEL_MAP = {"Cat": "cat", "Dog": "dog"}
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash raw file bytes (dedup key)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_image_ok(path: Path) -> bool:
    """Return True if file is a readable image."""
    try:
        with Image.open(path) as img:
            img.verify()
        # reopen after verify
        with Image.open(path) as img:
            _ = img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def scan_and_dedup() -> List[Tuple[Path, str]]:
    """
    Scan RAW_DIR, validate images, deduplicate by raw bytes hash.
    Returns list of (path, label) where label in {'cat','dog'} and each md5 is unique.
    """
    seen: Dict[str, Tuple[Path, str]] = {}
    dup_count = 0
    bad_count = 0
    total_files = 0

    for raw_label in ["Cat", "Dog"]:
        label = LABEL_MAP[raw_label]
        label_dir = RAW_DIR / raw_label
        if not label_dir.exists():
            print(f"[ERROR] Missing directory: {label_dir}", file=sys.stderr)
            sys.exit(1)

        for p in label_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue

            total_files += 1

            if not is_image_ok(p):
                bad_count += 1
                continue

            h = file_md5(p)
            if h in seen:
                dup_count += 1
                continue

            seen[h] = (p, label)

    samples = list(seen.values())
    print(f"[INFO] Raw files scanned: {total_files}")
    print(f"[INFO] Corrupted/unreadable skipped: {bad_count}")
    print(f"[INFO] Duplicates removed (raw-byte hash): {dup_count}")
    print(f"[INFO] Unique usable images: {len(samples)}")
    return samples


def stratified_split(samples: List[Tuple[Path, str]]) -> Dict[str, List[Tuple[Path, str]]]:
    """Stratified split by label with fixed SEED."""
    random.seed(SEED)
    by_class: Dict[str, List[Path]] = {"cat": [], "dog": []}
    for p, lbl in samples:
        by_class[lbl].append(p)

    for lbl in by_class:
        random.shuffle(by_class[lbl])

    def split_list(paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(paths)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        train = paths[:n_train]
        val = paths[n_train:n_train + n_val]
        test = paths[n_train + n_val:]
        return train, val, test

    out = {"train": [], "val": [], "test": []}
    for lbl, paths in by_class.items():
        tr, va, te = split_list(paths)
        out["train"].extend([(p, lbl) for p in tr])
        out["val"].extend([(p, lbl) for p in va])
        out["test"].extend([(p, lbl) for p in te])
        print(f"[INFO] Class '{lbl}': total={len(paths)}, train={len(tr)}, val={len(va)}, test={len(te)}")

    return out


def prepare_out_dirs():
    """Recreate output directory structure."""
    if OUT_DIR.exists():
        print(f"[INFO] Removing existing: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    for split in ["train", "val", "test"]:
        for lbl in ["cat", "dog"]:
            (OUT_DIR / split / lbl).mkdir(parents=True, exist_ok=True)


def process_and_save(src: Path, dst: Path):
    """Convert to RGB, resize, save as JPEG."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        img.save(dst, format="JPEG", quality=95)


def main():
    # Validate split sum
    if abs(sum(SPLITS.values()) - 1.0) > 1e-6:
        print("[ERROR] SPLITS must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Dedup+scan under: {RAW_DIR}")
    samples = scan_and_dedup()
    if not samples:
        print("[ERROR] No usable images after dedup.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Creating stratified splits...")
    split_data = stratified_split(samples)

    print(f"[INFO] Writing processed images to: {OUT_DIR} (size={IMG_SIZE}, RGB)")
    prepare_out_dirs()

    # Stable naming within each split/class
    counters = {("train", "cat"): 0, ("train", "dog"): 0,
                ("val", "cat"): 0, ("val", "dog"): 0,
                ("test", "cat"): 0, ("test", "dog"): 0}

    fail_count = 0
    for split_name, items in split_data.items():
        for src, lbl in items:
            counters[(split_name, lbl)] += 1
            idx = counters[(split_name, lbl)]
            dst = OUT_DIR / split_name / lbl / f"{lbl}_{idx:06d}.jpg"
            try:
                process_and_save(src, dst)
            except Exception as e:
                fail_count += 1
                print(f"[WARN] Failed {src}: {e}", file=sys.stderr)

    # Summary
    for split in ["train", "val", "test"]:
        cat_n = len(list((OUT_DIR / split / "cat").glob("*.jpg")))
        dog_n = len(list((OUT_DIR / split / "dog").glob("*.jpg")))
        print(f"[DONE] {split}: cat={cat_n}, dog={dog_n}, total={cat_n + dog_n}")

    if fail_count:
        print(f"[WARN] Failed to process {fail_count} images (skipped).")

    print("[SUCCESS] Dedup + preprocessing completed.")


if __name__ == "__main__":
    main()