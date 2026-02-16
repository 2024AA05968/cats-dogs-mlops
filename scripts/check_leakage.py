from pathlib import Path
import hashlib

ROOT = Path("data/processed")

def md5(path: Path, chunk=1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def collect(split: str) -> dict:
    # ✅ FIX: wrap the path expression before calling glob, then list()
    files = list((ROOT / split).glob("**/*.jpg"))
    return {md5(p): str(p) for p in files}

def main():
    train = collect("train")
    val = collect("val")
    test = collect("test")

    tv = set(train) & set(val)
    tt = set(train) & set(test)
    vt = set(val) & set(test)

    if tv or tt or vt:
        print("❌ Leakage detected!")
        if tv:
            print(f"train∩val: {len(tv)}")
            # print one example
            h = next(iter(tv))
            print("example:", train[h], "<->", val[h])
        if tt:
            print(f"train∩test: {len(tt)}")
            h = next(iter(tt))
            print("example:", train[h], "<->", test[h])
        if vt:
            print(f"val∩test: {len(vt)}")
            h = next(iter(vt))
            print("example:", val[h], "<->", test[h])
    else:
        print("✅ No leakage detected (no identical files across splits)")

if __name__ == "__main__":
    main()