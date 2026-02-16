import numpy as np
from PIL import Image
import io

from src.mlops_catsdogs.infer_utils import load_and_preprocess_image_bytes


def test_load_and_preprocess_image_bytes_shape_and_range():
    # Create a synthetic RGB image (random noise)
    arr = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    x = load_and_preprocess_image_bytes(image_bytes, img_size=224)

    # Expect shape [1, 3, 224, 224]
    assert tuple(x.shape) == (1, 3, 224, 224)

    # Tensor should be float in [0,1]
    assert x.dtype.is_floating_point
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.0