import json
import socket
import signal
import threading
from typing import Optional

from ..infrastructure.depth_anything_depth_estimator import DepthAnythingDepthEstimator
from ..use_cases.process_frame import process_frame
from ..utils.load_model import load_model


_shutdown_event = threading.Event()
_cache = {}


def _signal_handler(signum: int, _):
    signal_name = signal.Signals(signum).name
    print(f"\n🛑 Получен сигнал {signal_name}. Останавливаю сервер...")
    _shutdown_event.set()

def run_command_server_udp(host: str = "0.0.0.0", port: int = 9999):
    signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, _signal_handler) # kill, docker stop и т.д.

    try:
        bridge_detection_model = load_model("bridge_weights.pt")
        arch_gap_detection_model = load_model("arch_void_weights.pt")
        depth_estimator = DepthAnythingDepthEstimator()
        print("🧠 Модель загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((host, port))
        print(f"🚀 UDP API запущен на {host}:{port}")

        while not _shutdown_event.is_set():
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(65507)
                print(f"📩 Получено {len(data)} байт от {addr}")
                try:
                    packet = json.loads(data.decode('utf-8'))
                    b64_image = packet.get("frame")
                except Exception as e:
                    response = {"error": "Invalid JSON format",
                                "details": str(e)}
                else:
                    if not b64_image:
                        return json.dumps({"error": "No 'frame' in request"}).encode()
                    try:
                        response = process_frame(
                            b64_image=b64_image,
                            bridge_detector=bridge_detection_model,
                            arch_gap_detector=arch_gap_detection_model,
                            depth_estimator=depth_estimator,
                            cache=_cache
                        )
                    except Exception as e:
                        response = {"error": "Processing failed",
                                    "details": str(e)}
                
                response_bytes = json.dumps(
                    response,
                    ensure_ascii=False,
                    separators=(',', ':')
                ).encode('utf-8')
                sock.sendto(response_bytes, addr)

            except socket.timeout:
                continue
            except Exception as e:
                if _shutdown_event.is_set():
                    break
                print(f"❌ Ошибка приёма: {e}")

    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")
    finally:
        if sock:
            sock.close()
        print("🛑 UDP сервер остановлен. Ресурсы освобождены.")
        _shutdown_event.clear()
