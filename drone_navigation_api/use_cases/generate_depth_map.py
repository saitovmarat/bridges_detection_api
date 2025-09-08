from flask import Response, request, abort
import cv2
import numpy as np

from ..infrastructure.depth_estimator import DepthEstimator
from ..utils.depth_map_utils import normalize_and_convert_to_uint16


def get_depth_map_from_image(
    image: np.ndarray,
    depth_estimator: DepthEstimator
) -> np.ndarray:
    depth_map = depth_estimator.estimate(image)
    depth_map = normalize_and_convert_to_uint16(depth_map)
    return depth_map
