import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.mlops_catsdogs.infer_utils import predict_proba_from_pil


class DummyModel(nn.Module):
    """A tiny deterministic model for unit testing."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 2)
        # Make deterministic weights
        with torch.no_grad():
            self.fc.weight[:] = torch.tensor([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]])
            self.fc.bias[:] = 0.0

    def forward(self, x):
        # x: [B,3,H,W] -> [B,3]
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)


def test_predict_proba_from_pil_probabilities_sum_to_one():
    # Synthetic image
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    model = DummyModel()
    class_to_idx = {"cat": 0, "dog": 1}

    label, prob_dict = predict_proba_from_pil(
        model=model,
        img=img,
        class_to_idx=class_to_idx,
        img_size=224,
        device="cpu",
    )

    assert label in {"cat", "dog"}
    assert set(prob_dict.keys()) == {"cat", "dog"}

    s = prob_dict["cat"] + prob_dict["dog"]
    assert abs(s - 1.0) < 1e-5