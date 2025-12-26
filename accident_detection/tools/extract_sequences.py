from __future__ import annotations

"""Convert labeled videos into fixed-length frame sequences.

Beginner-friendly pipeline:
1) Put raw videos into:
      Datasets/raw/accident/
      Datasets/raw/no_accident/
2) Run this script to create sequences of frames in:
      Datasets/sequences/accident/<video>__seq000/000.jpg ...
      Datasets/sequences/no_accident/<video>__seq000/000.jpg ...
3) The script writes an index CSV (Datasets/index.csv) used by train.py.

Why sequences?
- A CNN processes each frame (spatial features)
- An LSTM processes the time dimension (motion / temporal patterns)

Performance tips (CCTV-friendly):
- We downsample to a target FPS (default 10) to reduce compute.
- We resize frames (default 112x112) to keep training fast.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


LABELS = ["accident", "no_accident"]


@dataclass(frozen=True)
class ExtractConfig:
    seq_len: int = 16
    stride: int = 8
    target_fps: float = 10.0
    image_size: int = 112
    max_sequences_per_video: int | None = None


def _iter_videos(root: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_video_to_sequences(video_path: Path, out_label_dir: Path, config: ExtractConfig) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Sample every Nth frame so we roughly match the desired target_fps.
    step = max(1, int(round(src_fps / config.target_fps)))

    frames: list[cv2.Mat] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            frame_resized = cv2.resize(frame, (config.image_size, config.image_size))
            frames.append(frame_resized)
        frame_index += 1

    cap.release()

    seq_dirs: list[Path] = []
    if len(frames) < config.seq_len:
        return seq_dirs

    base = video_path.stem
    seq_id = 0

    for start in range(0, len(frames) - config.seq_len + 1, config.stride):
        if config.max_sequences_per_video is not None and seq_id >= config.max_sequences_per_video:
            break

        seq_dir = out_label_dir / f"{base}__seq{seq_id:03d}"
        _safe_mkdir(seq_dir)

        for i in range(config.seq_len):
            frame = frames[start + i]
            # Save as numbered JPGs so the dataset loader can read them in sorted order.
            cv2.imwrite(str(seq_dir / f"{i:03d}.jpg"), frame)

        seq_dirs.append(seq_dir)
        seq_id += 1

    return seq_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frame sequences from labeled videos")
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "Datasets"),
        help="Path to Datasets/ folder",
    )
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--target_fps", type=float, default=10.0)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--max_sequences_per_video", type=int, default=0, help="0 means unlimited")

    args = parser.parse_args()
    datasets_dir = Path(args.datasets_dir)
    raw_dir = datasets_dir / "raw"
    out_dir = datasets_dir / "sequences"

    cfg = ExtractConfig(
        seq_len=args.seq_len,
        stride=args.stride,
        target_fps=args.target_fps,
        image_size=args.image_size,
        max_sequences_per_video=None if args.max_sequences_per_video == 0 else args.max_sequences_per_video,
    )

    # Ensure output folders exist.
    for label in LABELS:
        _safe_mkdir(out_dir / label)

    index_csv = datasets_dir / "index.csv"
    rows: list[str] = ["sequence_dir,label,video,seq_len,created_at"]

    for label in LABELS:
        label_in = raw_dir / label
        label_out = out_dir / label
        _safe_mkdir(label_in)

        videos = _iter_videos(label_in)
        for video in videos:
            seq_dirs = extract_video_to_sequences(video, label_out, cfg)
            for seq_dir in seq_dirs:
                rows.append(
                    f"{seq_dir.as_posix()},{label},{video.name},{cfg.seq_len},{datetime.utcnow().isoformat()}Z"
                )

    index_csv.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote index: {index_csv}")
    print("Next: run train.py to train CNN+LSTM.")


if __name__ == "__main__":
    main()