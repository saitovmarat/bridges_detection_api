import numpy as np
import torch
import cv2

from ..domain.depth_estimator_interface import DepthEstimatorInterface


class MidasDepthEstimator(DepthEstimatorInterface):
    def __init__(self, model_name="MiDaS_small", device="cpu"):
        self.device = torch.device(device)
        self.model_name = model_name

        self.model = torch.hub.load("isl-org/MiDaS", model_name)
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore

        if model_name == "MiDaS_small":
            self.transform = torch.hub.load(
                "isl-org/MiDaS", "transforms").small_transform  # type: ignore
        else:
            self.transform = torch.hub.load(
                "isl-org/MiDaS", "transforms").dpt_transform   # type: ignore

    def estimate(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)  # type: ignore

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            prediction = prediction.astype(np.float32)
            depth_map = cv2.bilateralFilter(
                prediction, d=9, sigmaColor=75, sigmaSpace=75)

        return depth_map
