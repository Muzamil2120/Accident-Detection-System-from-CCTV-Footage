from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from collections import deque


def iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def center_xywh(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


@dataclass
class Track:
    track_id: int
    bbox_xywh: tuple[int, int, int, int]
    score: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    # Simple kinematics (pixels / frame)
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0

    # Keep some history to estimate speed drop robustly.
    speed_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def update(self, new_bbox: tuple[int, int, int, int], score: float) -> None:
        old_cx, old_cy = center_xywh(self.bbox_xywh)
        new_cx, new_cy = center_xywh(new_bbox)

        self.vx = new_cx - old_cx
        self.vy = new_cy - old_cy
        self.speed = math.hypot(self.vx, self.vy)
        self.speed_history.append(self.speed)

        self.bbox_xywh = new_bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.age += 1


class SimpleByteTracker:
    """A light ByteTrack-style tracker (no ReID) using IOU association.

    Why this helps:
    - Tracking gives stable IDs; we can measure speed change per vehicle.
    - It reduces false positives from slow traffic because we require a sudden
      speed drop of a *tracked* vehicle, not just noisy per-frame boxes.

    Notes:
    - This is NOT full ByteTrack; it is a pragmatic approximation for a simple demo.
    - No extra dependencies (keeps the project easy to run).
    """

    _id_gen = itertools.count(1)

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_age: int = 20,
        min_hits: int = 3,
        high_conf: float = 0.50,
        low_conf: float = 0.30,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf = high_conf
        self.low_conf = low_conf

        self.tracks: list[Track] = []

    def _match(self, tracks: list[Track], dets: list[tuple[tuple[int, int, int, int], float]]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Greedy IOU matcher. Returns matches + unmatched indices."""
        if not tracks or not dets:
            return [], set(range(len(tracks))), set(range(len(dets)))

        candidates: list[tuple[float, int, int]] = []
        for ti, track in enumerate(tracks):
            for di, (bbox, _) in enumerate(dets):
                candidates.append((iou_xywh(track.bbox_xywh, bbox), ti, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        matches: list[tuple[int, int]] = []

        for score, ti, di in candidates:
            if score < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            matches.append((ti, di))

        unmatched_tracks = set(range(len(tracks))) - matched_tracks
        unmatched_dets = set(range(len(dets))) - matched_dets
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: list[tuple[tuple[int, int, int, int], float]]) -> list[Track]:
        """Update tracker and return *active* tracks."""

        high = [(bbox, score) for bbox, score in detections if score >= self.high_conf]
        low = [(bbox, score) for bbox, score in detections if self.low_conf <= score < self.high_conf]

        # 1) First associate high confidence detections.
        matches, unmatched_tracks, unmatched_high = self._match(self.tracks, high)
        for ti, di in matches:
            bbox, score = high[di]
            self.tracks[ti].update(bbox, score)

        # Mark unmatched tracks as missed for now.
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # 2) Optionally associate remaining tracks with low conf detections (ByteTrack idea).
        remaining_tracks = [t for t in self.tracks if t.time_since_update > 0]
        if remaining_tracks and low:
            # Map indices back to self.tracks
            remaining_map = [self.tracks.index(t) for t in remaining_tracks]
            rem_matches, rem_unmatched_tracks, rem_unmatched_low = self._match(remaining_tracks, low)
            for rti, di in rem_matches:
                bbox, score = low[di]
                self.tracks[remaining_map[rti]].update(bbox, score)

        # 3) Create new tracks for unmatched high detections.
        for di in unmatched_high:
            bbox, score = high[di]
            self.tracks.append(Track(track_id=next(self._id_gen), bbox_xywh=bbox, score=score))

        # 4) Prune dead tracks.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return tracks that are stable enough.
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]
