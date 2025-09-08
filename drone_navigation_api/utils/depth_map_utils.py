import numpy as np


def normalize_and_convert_to_uint16(
    depth_map: np.ndarray
) -> np.ndarray:
    if not isinstance(depth_map, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if depth_map.dtype == np.uint16:
        return depth_map

    print(f"⚠️ [CONVERT] Converting from {depth_map.dtype} to uint16...")

    depth_min = depth_map.min()
    depth_max = depth_map.max()

    if depth_max - depth_min > 0:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_map = np.zeros_like(depth_map)

    depth_map = (depth_map * 65535).astype(np.uint16)
    return depth_map
