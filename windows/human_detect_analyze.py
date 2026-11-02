import argparse
import base64
import threading
import time

import pyttsx3
import socketio
from openai import OpenAI

BOARD_URL = "http://192.168.1.71:7000"
DOWNSCALED_PATH = r"C:\Users\qiaos\VLM\detect_downscaled.jpg"
#DOWNSCALED_PATH = r"C:\Users\qiaos\VLM\senior-sit-on-ground.jpg"
DOWNSCALED_PATH_FALL = r"C:\Users\qiaos\VLM\detect_downscaled.jpg"
#DOWNSCALED_PATH_FALL = r"C:\Users\qiaos\VLM\samples\senior-sit-on-ground.jpg"
READ_INTERVAL_SEC = 3.0  # board pushes every ~3s; only act on one roughly every this many seconds
VLM_MODEL = "qwen3vl-flash:4b"
PROMPT = (
    "Analyze the person in the image step-by-step according to these strict rules and the strict order.\n "
    #"Output ONLY the final matching string. Do not include any explanation.\n"
    "Once a rule triggers a stop condition, that is your final classification.\n\n"
    "Rules:\n"
    "1. If the person in the image is in pain, output 'In pain' and stop.\n"
    "2. If the person doesn't stand up firmly and not sit on chair or sofa, output 'Fall' and stop.\n"
    "3. If the person is on the ground and not exercising, output 'Fall' and stop.\n"
    "4. Otherwise, output 'All Good'. \n"
     
)

# FastFlowLM's local OpenAI-compatible endpoint
client = OpenAI(
    base_url="http://localhost:52625/v1",
    api_key="not-needed-for-local"
)

sio = socketio.Client()

# Guards against overlapping ollama.generate() calls: if a new snapshot
# arrives while the previous one is still being analyzed, it's skipped
# rather than queued, so results stay close to real-time instead of
# backlogging further and further behind.
busy_lock = threading.Lock()
last_accepted_at = 0.0
busy_notice_printed = False  # only print the "still analyzing" notice once per busy stretch, not per skipped frame

def ask_vlm(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        stream=False,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


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
        with open(DOWNSCALED_PATH, "wb") as f:
            f.write(raw_image_bytes)

        vlm_start = time.time()
        result = ask_vlm(raw_image_bytes)
        vlm_delay = time.time() - vlm_start

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


def analyze_from_file(path):
    with open(path, "rb") as f:
        image_bytes = f.read()

    vlm_start = time.time()
    result = ask_vlm(image_bytes)
    vlm_delay = time.time() - vlm_start

    print(f"({vlm_delay:.2f}s) {result}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", action="store_true",
        help="Analyze the local image at DOWNSCALED_PATH_FALL instead of connecting to the board, then exit"
    )
    args = parser.parse_args()

    if args.n:
        analyze_from_file(DOWNSCALED_PATH_FALL)
    else:
        sio.connect(BOARD_URL)
        sio.wait()
