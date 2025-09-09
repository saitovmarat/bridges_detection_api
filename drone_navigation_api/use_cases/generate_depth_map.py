from flask import Response, request, abort
import cv2
import numpy as np

from ..domain.depth_estimator_interface import DepthEstimatorInterface
from ..utils.depth_map_utils import normalize_and_convert_to_uint16, \
                                    get_colorized_depth_map


def get_depth_map_from_image(
    image: np.ndarray,
    depth_estimator: DepthEstimatorInterface
) -> np.ndarray:
    depth_map = depth_estimator.estimate(image)
    depth_map_uint16 = normalize_and_convert_to_uint16(depth_map)
    depth_map_colorized = get_colorized_depth_map(depth_map_uint16)
    return depth_map_colorized
