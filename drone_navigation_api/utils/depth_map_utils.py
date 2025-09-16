import numpy as np
import cv2


def normalize_and_convert_to_uint16(
    depth_map: np.ndarray
) -> np.ndarray:
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if depth_map.dtype == np.uint16:
        return depth_map

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    if depth_max - depth_min > 0:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_map = np.zeros_like(depth_map)

    depth_map = (depth_map * 65535).astype(np.uint16)
    return depth_map

def get_colorized_depth_map(
    depth_map: np.ndarray
) -> np.ndarray:
    if depth_map.max() == depth_map.min():
        depth_map_normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    depth_map_colorized = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    return depth_map_colorized
