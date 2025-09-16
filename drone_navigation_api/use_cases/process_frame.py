import io
import cv2
import base64
import numpy as np
from typing import Any, Dict

from .get_target_point import get_target_point
from ..domain.detector_interface import DetectorInterface
from ..domain.depth_estimator_interface import DepthEstimatorInterface
from ..utils.is_bridge_confirmed import is_bridge_confirmed
from ..utils.is_tracking_lost import is_tracking_lost
from ..utils.update_object_confirmation import update_object_confirmation
from ..utils.reset_confirmation_state import reset_confirmation_state


def process_frame(
    b64_image: str,
    bridge_detector: DetectorInterface,
    arch_gap_detector: DetectorInterface,
    depth_estimator: DepthEstimatorInterface,
    cache: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        image_bytes = base64.b64decode(b64_image, validate=True)
        image_stream = io.BytesIO(image_bytes)
        image = cv2.imdecode(np.frombuffer(
            image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Invalid image format", "details": "Failed to decode base64 or image data"}
    except Exception as e:
        return {"error": "Invalid image format", "details": str(e)}

    try:
        bridge_detections = bridge_detector.detect(image)
        update_object_confirmation("bridge", bridge_detections, cache)

        arch_gap_detections = arch_gap_detector.detect(image)
        update_object_confirmation("arch_gap", arch_gap_detections, cache)

        target_point = None
        if arch_gap_detections:
            try:
                target_point = get_target_point(
                    image=image,
                    depth_estimator=depth_estimator,
                    arch_gap_detections=arch_gap_detections,
                    cache=cache
                )
            except Exception as e:
                print(f"⚠️ Ошибка расчёта target_point: {e}")

        if not is_bridge_confirmed(cache):
            status = "detecting"
            detections = [det.to_dict() for det in bridge_detections]
        elif is_tracking_lost("arch_gap", cache):
            status = "tracking_lost"
            detections = [det.to_dict() for det in arch_gap_detections]
        else:
            status = "bridge_confirmed"
            detections = [det.to_dict() for det in arch_gap_detections]

        result = {
            "status": status,
            "detections": detections,
            "target_point": target_point.to_dict() if target_point else {}
        }

    except Exception as e:
        return {"error": "Processing failed", "details": str(e)}

    return result
