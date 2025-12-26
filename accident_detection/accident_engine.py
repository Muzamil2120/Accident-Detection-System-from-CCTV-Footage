from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import base64
import itertools

import cv2
import numpy as np

from detector_yolo import YoloVehicleDetector
from tracking_bytetrack import SimpleByteTracker, Track, iou_xywh


@dataclass(frozen=True)
class EngineConfig:
    resize_width: int = 640
    frame_stride: int = 2
    max_frames: int = 800
    collision_iou: float = 0.35
    min_persist_frames: int = 5
    min_moving_speed_px: float = 8.0
    drop_ratio: float = 0.45
    motion_spike_threshold: float = 0.045


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class AccidentDetectionEngine:
    """Simple yet explained accident detector built on YOLO + ByteTrack."""

    def __init__(self, model_dir: Path, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.detector = YoloVehicleDetector(model_dir=model_dir)
        self.tracker = SimpleByteTracker(
            iou_threshold=0.30,
            max_age=20,
            min_hits=3,
            high_conf=0.50,
            low_conf=0.30,
        )

    def _resize_keep_aspect(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= self.config.resize_width:
            return frame, 1.0
        scale = self.config.resize_width / float(w)
        resized = cv2.resize(frame, (self.config.resize_width, int(h * scale)))
        return resized, scale

    def _motion_score(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
        if prev_gray is None:
            return 0.0
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.mean(diff)) / 255.0

    def _sudden_speed_drop(self, track: Track) -> bool:
        if len(track.speed_history) < 8:
            return False
        speeds = list(track.speed_history)
        median_prev = float(np.median(speeds[:-1]))
        current = speeds[-1]
        if median_prev < self.config.min_moving_speed_px:
            return False
        return current < median_prev * self.config.drop_ratio

    def _estimate_confidence(self, signal_count: int, persist: int) -> float:
        base = 0.25 + 0.25 * min(3, signal_count) + 0.05 * min(10, persist)
        return round(min(0.98, base), 2)

    def _serialize_tracks(self, tracks: list[Track]) -> list[dict]:
        return [
            {
                "track_id": track.track_id,
                "bbox": [int(v) for v in track.bbox_xywh],
                "score": round(track.score, 3),
                "speed": round(track.speed, 2),
            }
            for track in tracks
        ]

    def _annotate_frame(
        self, frame: np.ndarray | None, tracks: list[Track], highlight_pair: tuple[int, int] | None
    ) -> str | None:
        if frame is None:
            return None
        annotated = frame.copy()
        pair_ids = set(highlight_pair or ())
        for track in tracks:
            x, y, w, h = track.bbox_xywh
            color = (0, 220, 32) if track.track_id not in pair_ids else (20, 20, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2, lineType=cv2.LINE_AA)
            label = f"#{track.track_id} {track.score:.2f}"
            cv2.putText(
                annotated,
                label,
                (x, max(y - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            return None
        return base64.b64encode(buffer).decode("ascii")

    def _build_response(
        self,
        accident: bool,
        confidence: float,
        frames_processed: int,
        persist: int,
        tracks: list[Track],
        pair: tuple[int, int] | None,
        preview: str | None,
    ) -> dict:
        return {
            "accident": accident,
            "confidence": confidence,
            "timestamp": _now_iso(),
            "frames_processed": frames_processed,
            "persistence": persist,
            "vehicle_count": len(tracks),
            "preview_image": preview,
            "tracks": self._serialize_tracks(tracks),
            "overlap_pair": list(pair) if pair else None,
        }

    def detect_video(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Unable to read the uploaded video file.")

        prev_gray = None
        persist = 0
        best_conf = 0.0
        processed = 0
        frame_index = 0
        last_frame: np.ndarray | None = None
        last_tracks: list[Track] = []
        last_pair: tuple[int, int] | None = None
        active_pair: tuple[int, int] | None = None

        while processed < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % self.config.frame_stride != 0:
                continue
            processed += 1

            resized, _ = self._resize_keep_aspect(frame)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            motion = self._motion_score(prev_gray, gray)
            motion_spike = motion >= self.config.motion_spike_threshold

            detections = self.detector.detect(resized)
            tracks = self.tracker.update([(d.bbox_xywh, d.score) for d in detections])

            last_frame = resized.copy()
            last_tracks = list(tracks)
            prev_gray = gray

            if len(tracks) < 2:
                persist = 0
                active_pair = None
                continue

            best = (0.0, None)
            for t1, t2 in itertools.combinations(tracks, 2):
                overlap = iou_xywh(t1.bbox_xywh, t2.bbox_xywh)
                if overlap > best[0]:
                    best = (overlap, (t1, t2))

            overlap, pair = best
            collision = overlap >= self.config.collision_iou
            speed_drop = False
            if pair:
                speed_drop = self._sudden_speed_drop(pair[0]) or self._sudden_speed_drop(pair[1])

            signal_count = sum((collision, speed_drop, motion_spike))

            if collision and signal_count < 2:
                persist = 0
                active_pair = None
                continue

            if pair and collision and signal_count >= 2:
                id1, id2 = sorted((pair[0].track_id, pair[1].track_id))
                pair_key = (id1, id2)
                if active_pair == pair_key:
                    persist += 1
                else:
                    active_pair = pair_key
                    persist = 1
                last_pair = active_pair
            else:
                persist = 0
                active_pair = None
                last_pair = None

            if persist > 0:
                conf = self._estimate_confidence(signal_count, persist)
                best_conf = max(best_conf, conf)
            else:
                conf = 0.0

            if persist >= self.config.min_persist_frames:
                preview_image = self._annotate_frame(last_frame, last_tracks, last_pair)
                cap.release()
                return self._build_response(True, conf, processed, persist, last_tracks, last_pair, preview_image)

        cap.release()
        preview_image = self._annotate_frame(last_frame, last_tracks, last_pair)
        final_conf = round(min(0.7, best_conf), 2)
        return self._build_response(False, final_conf, processed, persist, last_tracks, last_pair, preview_image)from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import itertools

import cv2
import numpy as np

from detector_yolo import YoloVehicleDetector
from tracking_bytetrack import SimpleByteTracker, Track, iou_xywh


@dataclass
class EngineConfig:
    # FPS / real-time optimization
    resize_width: int = 640
    frame_stride: int = 2  # process every Nth frame (2 => ~half load)
    max_frames: int = 900

    # Accident heuristics (conservative to reduce false positives)
    collision_iou: float = 0.35
    min_persist_frames: int = 5

    # Speed-drop gating: ignores slow traffic stops
    min_moving_speed_px: float = 8.0
    drop_ratio: float = 0.45

    # Motion intensity (frame difference) for impact-like spikes
    motion_spike_threshold: float = 0.045


class AccidentDetectionEngine:
    """Multi-signal accident detection using detection + tracking.

    Signals used together:
    1) YOLO vehicle detection
    2) IoU overlap between two tracked vehicles
    3) Sudden stop / speed drop of (at least) one tracked vehicle
    4) Motion intensity spike (frame difference) suggesting impact
    5) Persistence across multiple consecutive processed frames

    Performance improvements:
    - Resize before inference: fewer pixels => faster DNN forward.
    - Frame stride: process every Nth frame to match real CCTV FPS.
    - NMS in YOLO detector reduces duplicate boxes => fewer expensive pair checks.
    """

    def __init__(self, model_dir: Path, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.detector = YoloVehicleDetector(model_dir=model_dir)
        self.tracker = SimpleByteTracker(
            iou_threshold=0.30,
            max_age=20,
            min_hits=3,
            high_conf=0.50,
            low_conf=0.30,
        )

    def _resize_keep_aspect(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= self.config.resize_width:
            return frame, 1.0
        scale = self.config.resize_width / float(w)
        resized = cv2.resize(frame, (self.config.resize_width, int(h * scale)))
        return resized, scale

    def _motion_score(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
        if prev_gray is None:
            return 0.0
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.mean(diff)) / 255.0

    def _sudden_speed_drop(self, track: Track) -> bool:
        # Require enough history to avoid noise.
        if len(track.speed_history) < 8:
            return False

        speeds = list(track.speed_history)
        median_prev = float(np.median(speeds[:-1]))
        current = speeds[-1]

        # Ignore slow traffic: only consider drops from a reasonably moving state.
        if median_prev < self.config.min_moving_speed_px:
            return False

        return current < median_prev * self.config.drop_ratio

    def detect_video(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Unable to read the uploaded video file.")

        prev_gray = None
        persist = 0
        best_conf = 0.0

        # Track collision pairs over time by their IDs.
        active_pair = None  # (id1, id2)

        processed = 0
        frame_index = 0

        while processed < self.config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if frame_index % self.config.frame_stride != 0:
                continue

            processed += 1

            resized, scale = self._resize_keep_aspect(frame)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            motion = self._motion_score(prev_gray, gray)
            motion_spike = motion >= self.config.motion_spike_threshold

            dets = self.detector.detect(resized)

            # Scale bboxes back to original resized coordinates (we already detect on resized)
            detections = [(d.bbox_xywh, d.score) for d in dets]
            tracks = self.tracker.update(detections)

            # No tracks => reset persistence.
            if len(tracks) < 2:
                persist = 0
                active_pair = None
                prev_gray = gray
                continue

            # Find best overlapping pair (highest IoU) between *tracked* vehicles.
            best = (0.0, None)
            for t1, t2 in itertools.combinations(tracks, 2):
                overlap = iou_xywh(t1.bbox_xywh, t2.bbox_xywh)
                if overlap > best[0]:
                    best = (overlap, (t1, t2))

            overlap, pair = best
            collision = overlap >= self.config.collision_iou

            speed_drop = False
            if pair:
                speed_drop = self._sudden_speed_drop(pair[0]) or self._sudden_speed_drop(pair[1])

            # Multi-signal rule: require at least 2/3 signals with persistence.
            signal_count = sum((collision, speed_drop, motion_spike))

            # Also reduce slow-traffic false positives:
            # - If vehicles overlap but no speed drop and no motion spike => reset.
            if collision and signal_count < 2:
                persist = 0
                active_pair = None
                prev_gray = gray
                continue

            if pair and collision and signal_count >= 2:
                id1, id2 = sorted((pair[0].track_id, pair[1].track_id))
                pair_key = (id1, id2)
                if active_pair == pair_key:
                    persist += 1
                else:
                    active_pair = pair_key
                    persist = 1
            else:
                persist = 0
                active_pair = None

            # Confidence: conservative; grows with persistence and signal strength.
            if persist > 0:
                conf = 0.25 + 0.25 * min(3, signal_count) + 0.05 * min(10, persist)
                best_conf = max(best_conf, conf)

            if persist >= self.config.min_persist_frames:
                cap.release()
                return {
                    "accident": True,
                    "confidence": round(min(0.98, best_conf), 2),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

            prev_gray = gray

        cap.release()
        return {
            "accident": False,
            "confidence": round(min(0.7, best_conf), 2),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
