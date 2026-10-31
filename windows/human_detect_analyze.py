import threading
import time
from io import BytesIO

import ollama
import pyttsx3
import socketio
from PIL import Image

BOARD_URL = "http://192.168.1.71:7000"
DOWNSCALED_PATH = r"C:\Users\qiaos\VLM\detect_downscaled.jpg"
DOWNSCALED_PATH_FALL = r"C:\Users\qiaos\VLM\senior_fall.jpg"
DOWNSCALED_PATH_PAIN = r"C:\Users\qiaos\VLM\senior_pain.jpg"
READ_INTERVAL_SEC = 15.0  # board pushes every ~10s; only act on one roughly every this many seconds
PROMPT = (
    'If the person upper body is not upright, answer "Fall" and quit'
    'If the person is on the ground, answer "Fall" and quit'
    'If the person in the image is in pain, answer "In pain" and quit'
    'Otherwise, answer "All Good"'
    #'Describe the person behavior less than 20 words"'
)

sio = socketio.Client()

# Guards against overlapping ollama.generate() calls: if a new snapshot
# arrives while the previous one is still being analyzed, it's skipped
# rather than queued, so results stay close to real-time instead of
# backlogging further and further behind.
busy_lock = threading.Lock()
last_accepted_at = 0.0
busy_notice_printed = False  # only print the "still analyzing" notice once per busy stretch, not per skipped frame

def speak_alert(message):
    # pyttsx3's SAPI5 driver on Windows only reliably speaks once per
    # engine instance -- a second say()/runAndWait() on a reused engine
    # silently does nothing. Create a fresh engine for every alert instead
    # of caching one.
    print(f"ALERT: {message}")
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()
    engine.stop()


@sio.event
def connect():
    print(f"Connected to {BOARD_URL}, waiting for detections...")


@sio.event
def disconnect():
    print("Disconnected.")


def analyze(raw_image_bytes, timestamp):
    try:
        image = Image.open(BytesIO(raw_image_bytes))
        # LANCZOS gives sharper downscaled detail than the default nearest-
        # neighbor resize; quality=100 and subsampling=0 match the original's
        # own quality=100 encoding so file size tracks the pixel-area
        # reduction instead of compounding with extra lossy compression.
        downscaled = image.resize(
            (image.width // 2, image.height // 2), Image.LANCZOS
        )
        buf = BytesIO()
        downscaled.save(buf, format="JPEG", quality=100, subsampling=0)
        downscaled_bytes = buf.getvalue()

        with open(DOWNSCALED_PATH, "wb") as f:
            f.write(downscaled_bytes)

        vlm_start = time.time()
        response = ollama.generate(
            model='qwen2.5vl:7b',
            prompt=PROMPT,
            images=[downscaled_bytes]  # Pass the downscaled image, not the original
        )
        vlm_delay = time.time() - vlm_start

        result = response["response"]
        print(f"[{timestamp}] ({vlm_delay:.2f}s) {result}")

        result_lower = result.lower()
        if "in pain" in result_lower:
            speak_alert("Alert, Mister Williams is in pain, please help")
        elif "fall" in result_lower:
            speak_alert("Alert, Mister Williams falls, please help")
    except Exception as e:
        print(f"[{timestamp}] Analysis failed: {e!r}")
    finally:
        busy_lock.release()


@sio.on("snapshot")
def on_snapshot(data):
    global last_accepted_at, busy_notice_printed

    # Throttle to roughly one read every READ_INTERVAL_SEC, even though the
    # board pushes more often -- snapshots arriving too soon are skipped
    # quietly, since this is routine pacing rather than a problem.
    now = time.time()
    if now - last_accepted_at < READ_INTERVAL_SEC:
        return

    # ollama.generate() can take well over the read interval (especially on
    # CPU) -- running it directly in this handler would block the same
    # thread that answers the websocket's ping/pong keepalives, risking a
    # dropped connection. start_background_task hands it off so this
    # handler returns immediately.
    if not busy_lock.acquire(blocking=False):
        if not busy_notice_printed:
            print(f"[{data.get('timestamp')}] Still analyzing previous snapshot -- skipping until done")
            busy_notice_printed = True
        return

    busy_notice_printed = False
    last_accepted_at = now
    sio.start_background_task(analyze, data["image"], data.get("timestamp"))


if __name__ == "__main__":
    sio.connect(BOARD_URL)
    sio.wait()
