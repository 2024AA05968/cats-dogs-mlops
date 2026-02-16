from __future__ import annotations

import io
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_and_preprocess_image_bytes(
    image_bytes: bytes,
    img_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess raw image bytes into a torch tensor of shape [1, 3, img_size, img_size].
    - Converts to RGB
    - Resizes
    - Converts to tensor
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return x


@torch.no_grad()
def predict_proba_from_pil(
    model: torch.nn.Module,
    img: Image.Image,
    class_to_idx: Dict[str, int],
    img_size: int = 224,
    device: str | torch.device = "cpu",
) -> Tuple[str, Dict[str, float]]:
    """
    Run inference for a PIL image:
    - preprocess
    - forward pass
    - softmax
    Returns (predicted_label, probabilities_dict)
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    prob_dict = {idx_to_class[i]: float(probs[i]) for i in range(len(probs))}
    pred_idx = int(torch.argmax(torch.tensor(probs)))
    pred_label = idx_to_class[pred_idx]

    return pred_label, prob_dict