from importlib.resources import files, as_file
from ..infrastructure.yolo_detector import YoloDetector


_model_cache = {}


def load_model(weights_filename: str) -> YoloDetector:
    if weights_filename in _model_cache:
        return _model_cache[weights_filename]

    ref = files("drone_navigation_api.assets") / weights_filename
    with as_file(ref) as temp_path:
        model = YoloDetector(str(temp_path))

    _model_cache[weights_filename] = model
    return model
