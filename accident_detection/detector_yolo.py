from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VEHICLE_CLASSES = {"car", "motorbike", "bus", "truck", "bicycle"}


@dataclass(frozen=True)
class Detection:
    """One detection output from YOLO."""

    bbox_xywh: tuple[int, int, int, int]
    score: float
    class_name: str


def _resolve_model_file(model_dir: Path, base_name: str) -> Path:
    """Resolve a model filename.

    Many users accidentally download `coco.names` as `coco.names.txt` on Windows.
    We accept both to reduce friction.
    """

    direct = model_dir / base_name
    if direct.exists():
        return direct

    txt = model_dir / f"{base_name}.txt"
    if txt.exists():
        return txt

    return direct


def resolve_yolo_paths(model_dir: Path) -> tuple[Path, Path, Path]:
    cfg = _resolve_model_file(model_dir, "yolov3-tiny.cfg")
    weights = _resolve_model_file(model_dir, "yolov3-tiny.weights")
    names = _resolve_model_file(model_dir, "coco.names")
    return cfg, weights, names


class YoloVehicleDetector:
    """YOLOv3-tiny vehicle detector using OpenCV DNN.

    Performance notes:
    - We use NMS to remove duplicate boxes (reduces false positives).
    - We resize frames for detection and scale boxes back.
    """

    def __init__(
        self,
        model_dir: Path,
        conf_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (416, 416),
    ) -> None:
        self.model_dir = model_dir
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        cfg, weights, names = resolve_yolo_paths(model_dir)
        self.cfg_path = cfg
        self.weights_path = weights
        self.names_path = names

        missing = [p.name for p in (cfg, weights, names) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing YOLO model file(s): {', '.join(missing)}. "
                f"Place them in {model_dir} (accepts optional .txt extension for cfg/names)."
            )

        with open(self.names_path, encoding="utf-8") as file:
            self.class_names = [line.strip() for line in file if line.strip()]

        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1 / 255.0,
            size=self.input_size,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                if score < self.conf_threshold:
                    continue

                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                if class_name not in VEHICLE_CLASSES:
                    continue

                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, cx - w // 2)
                y = max(0, cy - h // 2)

                boxes.append([x, y, w, h])
                scores.append(score)
                class_ids.append(class_id)

        if not boxes:
            return []

        # NMS reduces duplicate boxes which often cause artificial overlaps.
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        if len(keep) == 0:
            return []

        detections: list[Detection] = []
        for idx in keep.flatten().tolist():
            bbox = tuple(int(v) for v in boxes[idx])
            score = float(scores[idx])
            class_name = self.class_names[class_ids[idx]] if class_ids[idx] < len(self.class_names) else "unknown"
            detections.append(Detection(bbox_xywh=bbox, score=score, class_name=class_name))

        return detections
