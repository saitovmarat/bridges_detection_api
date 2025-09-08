import base64
from PIL import Image
from io import BytesIO
import numpy as np
import cv2


def decode_base64_image(b64_str: str) -> np.ndarray:
    try:
        if b64_str.startswith('data:image'):
            b64_str = b64_str.split(',')[1]

        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported image format")

        return image_np.astype(np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")
