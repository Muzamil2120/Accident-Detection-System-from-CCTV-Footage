from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from accident_engine import AccidentDetectionEngine, EngineConfig

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"mp4"}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

engine: AccidentDetectionEngine | None = None
engine_error: str | None = None
try:
    engine = AccidentDetectionEngine(
        model_dir=BASE_DIR / "models",
        config=EngineConfig(
            resize_width=640,
            frame_stride=2,
            collision_iou=0.35,
            min_persist_frames=5,
            min_moving_speed_px=8.0,
            drop_ratio=0.45,
            motion_spike_threshold=0.045,
        ),
    )
except Exception as exc:  # noqa: BLE001
    engine_error = str(exc)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        model_ready=engine is not None,
        model_error=engine_error,
    )


@app.get("/health")
def health() -> str:
    return jsonify(
        {
            "ok": True,
            "model_ready": engine is not None,
            "model_error": engine_error,
        }
    )


def detect_accident(video_path: Path) -> dict:
    if engine is None:
        raise RuntimeError(engine_error or "Model is not ready.")
    return engine.detect_video(video_path)


@app.post("/detect")
def handle_detection() -> tuple[dict, int]:
    if engine is None:
        return ({"success": False, "error": engine_error or "Detector unavailable."}, 503)

    uploaded_file = request.files.get("video")
    if not uploaded_file or uploaded_file.filename == "":
        return ({"success": False, "error": "Please attach an MP4 video."}, 400)
    if not allowed_file(uploaded_file.filename):
        return ({"success": False, "error": "Only MP4 files are supported."}, 400)

    timestamp = int(datetime.utcnow().timestamp() * 1000)
    safe_name = secure_filename(uploaded_file.filename)
    filename = f"{timestamp}_{safe_name}"
    destination = UPLOAD_FOLDER / filename
    uploaded_file.save(str(destination))

    try:
        detection = detect_accident(destination)
    except Exception as exc:  # noqa: BLE001
        return ({"success": False, "error": str(exc)}, 500)

    response = {
        "success": True,
        "message": "🚨 Accident Happened" if detection["accident"] else "✅ No Accident Detected",
        "accident": detection["accident"],
        "confidence": detection["confidence"],
        "timestamp": detection["timestamp"],
        "frames_processed": detection["frames_processed"],
        "persistence": detection["persistence"],
        "vehicle_count": detection["vehicle_count"],
        "preview_image": detection["preview_image"],
        "tracks": detection["tracks"],
        "overlap_pair": detection["overlap_pair"],
        "video_url": f"/static/uploads/{filename}",
    }
    return response, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from accident_engine import AccidentDetectionEngine, EngineConfig


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
MODEL_DIR = BASE_DIR / "models"
ALLOWED_EXTENSIONS = {"mp4"}

# Ensure key directories exist before the server runs
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

engine = None
engine_error = None
try:
    # Conservative defaults to reduce false positives on slow traffic CCTV.
    engine = AccidentDetectionEngine(
        model_dir=MODEL_DIR,
        config=EngineConfig(
            resize_width=640,
            frame_stride=2,
            collision_iou=0.35,
            min_persist_frames=5,
            min_moving_speed_px=8.0,
            drop_ratio=0.45,
            motion_spike_threshold=0.045,
        ),
    )
except Exception as exc:  # noqa: BLE001
    engine_error = str(exc)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


@app.get("/health")
def health():
    # Simple health endpoint useful for debugging fetch() failures.
    return jsonify({"ok": True, "model_ready": engine is not None, "model_error": engine_error})


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_accident(video_path: Path) -> dict:
    if engine is None:
        raise RuntimeError(engine_error or "Model is not ready.")
    return engine.detect_video(video_path)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None)

    uploaded_file = request.files.get("video")
    if not uploaded_file or uploaded_file.filename == "":
        return render_template("index.html", result="Please upload an MP4 video.")
    if not allowed_file(uploaded_file.filename):
        return render_template("index.html", result="Only MP4 files are supported.")

    filename = secure_filename(uploaded_file.filename)
    destination = UPLOAD_FOLDER / filename
    uploaded_file.save(str(destination))  # save for processing

    try:
        detection_result = detect_accident(destination)
        result_message = "🚨 Accident Happened" if detection_result["accident"] else "✅ No Accident"
        response_payload = {
            "accident": detection_result["accident"],
            "confidence": detection_result["confidence"],
            "timestamp": detection_result["timestamp"],
            "video": filename,
        }
    except Exception as error:
        result_message = f"Processing error: {error}"
        response_payload = {
            "accident": False,
            "confidence": 0.0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    if "application/json" in request.headers.get("Accept", ""):
        return jsonify(response_payload)

    return render_template("index.html", result=result_message)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
