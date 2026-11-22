
from flask import Flask, Response
import cv2
import threading
import time

app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock()

def update_frame(frame):
    global latest_jpeg
    frame = cv2.resize(frame, (640, 480))
    ok, jpeg = cv2.imencode(
        ".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if ok:
        with lock:
            latest_jpeg = jpeg.tobytes()

@app.route("/video")
def video():
    def generate():
        while True:
            with lock:
                data = latest_jpeg
            if data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + data + b"\r\n"
                )
            time.sleep(1 / 30)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def start_server():
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=5000,
            threaded=True,
            use_reloader=False
        ),
        daemon=True
    ).start()
