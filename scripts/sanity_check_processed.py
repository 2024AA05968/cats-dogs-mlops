from pathlib import Path
from PIL import Image
import random

ROOT = Path("data/processed")
SPLITS = ["train", "val", "test"]
CLASSES = ["cat", "dog"]

def sample_images(n=20):
    files = []
    for s in SPLITS:
        for c in CLASSES:
            files += list((ROOT / s / c).glob("*.jpg"))
    random.shuffle(files)
    return files[:n]

def main():
    bad = []
    for p in sample_images(30):
        try:
            with Image.open(p) as im:
                mode = im.mode
                size = im.size
            if size != (224, 224) or mode != "RGB":
                bad.append((str(p), size, mode))
        except Exception as e:
            bad.append((str(p), "ERR", str(e)))

    if bad:
        print("❌ Found issues:")
        for row in bad:
            print(row)
    else:
        print("✅ Sampled images look good: all are 224x224 RGB")

if __name__ == "__main__":
    main()