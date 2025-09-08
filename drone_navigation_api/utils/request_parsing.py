from flask import request

def get_frame_from_request(
    request: request
) -> str:
    data = request.get_json()
    b64_image = data.get("frame")
    return b64_image
