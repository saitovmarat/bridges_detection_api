import threading
import time
from flask import Blueprint, abort, request, Response
import cv2

from ..infrastructure.midas_depth_estimator import MidasDepthEstimator
from ..infrastructure.depth_anything_depth_estimator import DepthAnythingDepthEstimator
from ..use_cases.generate_depth_map import get_depth_map_from_image
from ..utils.request_parsing import get_frame_from_request
from ..utils.image_utils import decode_base64_image


_depth_estimator_instance = None
blueprint = Blueprint('depth_api', __name__, url_prefix='/depth')


def get_or_create_depth_estimator():
    global _depth_estimator_instance
    if _depth_estimator_instance is None:
        _depth_estimator_instance = DepthAnythingDepthEstimator()
    return _depth_estimator_instance

@blueprint.route('', methods=['POST'])
def get_depth_map():
    print(f"🧵 Threads during request: {threading.active_count()}")
    start_time = time.time()
    thread_id = threading.get_ident()
    print(f"🧵 [{thread_id}] Начал обработку запроса в {time.strftime('%H:%M:%S')}")

    try:
        depth_estimator = get_or_create_depth_estimator()
        b64_image = get_frame_from_request(request)
        image = decode_base64_image(b64_image)
        depth_map = get_depth_map_from_image(image, depth_estimator)
    except ValueError as e:
        abort(400, description=str(e))
    except Exception as e:
        print(f"❌ SERVER ERROR: {str(e)}") 
        abort(500, description=f"Depth estimation failed: {str(e)}")

    success, png_buffer = cv2.imencode('.png', depth_map)
    if not success:
        abort(500, description="Failed to encode depth map to PNG")

    print(f"🧵 [{thread_id}] Завершил за {time.time() - start_time:.2f} сек")
    return Response(
        png_buffer.tobytes(),
        mimetype='image/png',
        headers={
            'Content-Disposition': 'inline; filename="depth.png"',
            'X-Depth-Width': str(depth_map.shape[1]),
            'X-Depth-Height': str(depth_map.shape[0]),
        }
    )

@blueprint.route('/health', methods=['GET'])
def health():
    return {
        "status": "ok" if _depth_estimator_instance is not None else "error",
        "depth_estimator_loaded": _depth_estimator_instance is not None
    }
