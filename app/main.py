from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

# --- Windows: stabilize torch DLL loading (same approach as training) ---
if sys.platform.startswith("win"):
    try:
        import site
        sp = site.getsitepackages()[0]
        torch_lib = Path(sp) / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

import torch
import torch.nn.functional as F
from torchvision import transforms

from app.model import BaselineCNN

app = FastAPI(title="Cats vs Dogs Inference API", version="1.0")

MODEL_PATH = Path("models/baseline_cnn.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model: BaselineCNN | None = None
class_to_idx: Dict[str, int] | None = None
idx_to_class: Dict[int, str] | None = None

# Same resize used in training
IMG_SIZE = 224
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


@app.on_event("startup")
def load_model():
    global model, class_to_idx, idx_to_class

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}. Train and save the model first.")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ckpt.get("class_to_idx", {"cat": 0, "dog": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = BaselineCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[INFO] Model loaded from {MODEL_PATH} on {DEVICE}")
    print(f"[INFO] class_to_idx: {class_to_idx}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file and get cat/dog prediction with probabilities.
    """
    if model is None or idx_to_class is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Basic content-type check (not perfect but helpful)
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    x = preprocess(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)))
    label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            idx_to_class[0]: float(probs[0]),
            idx_to_class[1]: float(probs[1]),
        }
    }