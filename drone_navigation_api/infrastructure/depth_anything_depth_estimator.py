import numpy as np
import torch
import cv2
import os

from depth_anything_v2.dpt import DepthAnythingV2 
from ..domain.depth_estimator_interface import DepthEstimatorInterface


class DepthAnythingDepthEstimator(DepthEstimatorInterface):
    def __init__(self, model_path=None, device="cpu"):
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'assets', 'depth_anything_v2_vits.pth')
            model_path = os.path.normpath(model_path) 

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Couldn't load weights from': {model_path}")

        encoder = self._get_encoder_from_path(model_path)

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {encoder}. Choose from: {list(model_configs.keys())}")

        self.model = DepthAnythingV2(**model_configs[encoder])
        
        print(f"📂 Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        print(f"✅ Model {encoder} loaded successfully on {self.device}")

    def _get_encoder_from_path(self, model_path: str) -> str:
        filename = model_path.lower()
        for enc in ['vits', 'vitb', 'vitl', 'vitg']:
            if enc in filename:
                return enc
        raise ValueError(f"Cannot determine encoder from filename: {model_path}. Expected 'vits', 'vitb', 'vitl', or 'vitg'.")

    def estimate(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")

        with torch.no_grad():
            depth_map = self.model.infer_image(image)  

        depth_map = depth_map.astype(np.float32)

        #depth_map = cv2.bilateralFilter(
        #    depth_map,
        #    d=9,
        #    sigmaColor=75,
        #    sigmaSpace=75
        #)

        return depth_map
