import threading
import time

from fastapi import Response

from arduino.app_bricks.object_detection import ObjectDetection
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App
from arduino.app_utils import Logger
from arduino.app_utils.image import compress_to_jpeg

logger = Logger("human-detect-analysis")

CAPTURE_INTERVAL_SEC = 10.0
CONFIDENCE_THRESHOLD = 0.5

# Same camera setup as the working cameraCheck app: resolution/fps pinned,
# started once and kept open for the app's lifetime (re-creating the Camera
# object every cycle was confirmed in testing to make the device fail to
# reopen on the very next cycle).
camera = Camera(resolution=(1280, 720), fps=30)
camera.start()

detector = ObjectDetection()

# apps_checkout only mirrors the app's source tree (app.yaml, python/,
# sketch/, assets/) -- confirmed in testing that files written at runtime to
# any of those paths never show up in the checked-out mirror, because the
# running container's filesystem isn't bind-mounted back to the host for
# arbitrary writes. Serving the latest snapshot bytes over HTTP via this
# brick's own web server is the only way to get it off the board.
snapshot_lock = threading.Lock()
latest_snapshot = None  # most recent JPEG bytes with a person detected
latest_detected_at = None  # epoch seconds of that snapshot

ui = WebUI()


def get_snapshot():
    with snapshot_lock:
        data = latest_snapshot
    if data is None:
        return Response(status_code=404)
    return Response(content=data, media_type="image/jpeg")


def get_status():
    # Tiny JSON a browser can poll frequently without re-downloading the
    # image every time -- only fetch /snapshot when last_detected_at changes.
    with snapshot_lock:
        return {"last_detected_at": latest_detected_at}


ui.expose_api("GET", "/snapshot", get_snapshot)
ui.expose_api("GET", "/status", get_status)


def loop():
    global latest_snapshot, latest_detected_at

    # Discard a few frames so auto-exposure has settled -- same warm-up
    # cameraCheck uses before its real capture.
    for _ in range(5):
        camera.capture()
        time.sleep(0.1)

    frame = camera.capture()
    jpeg_bytes = compress_to_jpeg(frame=frame, quality=100).tobytes()

    results = detector.detect(jpeg_bytes, confidence=CONFIDENCE_THRESHOLD)
    detections = results.get("detection", []) if results else []
    found_person = any(d.get("class_name") == "person" for d in detections)

    if found_person:
        detected_at = time.time()
        with snapshot_lock:
            latest_snapshot = jpeg_bytes
            latest_detected_at = detected_at
        # Push to every connected Socket.IO client (e.g. a Python script using
        # the python-socketio client library) -- no polling needed on their end.
        ui.send_message("snapshot", {"timestamp": detected_at, "image": jpeg_bytes})
        logger.info("Person detected -- snapshot updated, available at /snapshot and pushed over websocket")
    else:
        logger.info("No person detected in this frame")

    time.sleep(CAPTURE_INTERVAL_SEC)


App.run(user_loop=loop)
