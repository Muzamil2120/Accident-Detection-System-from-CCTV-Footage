# CCTV Accident Detection (Simple)

This workspace contains a beginner-friendly CCTV accident detector powered by a pre-trained YOLOv3-tiny model and a lightweight ByteTrack-inspired tracker. The goal is to upload short `.mp4` clips, detect overlapping vehicles, and keep the whole stack easy to understand.

## Features
1. **Video upload UI** (`templates/index.html`): previews the clip, calls `/detect`, and shows annotated bounding boxes plus confidence.
2. **YOLO + tracker engine** (`accident_engine.py`): resizes frames, uses frame-stride for better FPS, filters slow traffic, detects sudden speed drops, and reports accidents only when overlap persists for at least five frames.
3. **JSON API** (`/detect`): returns accident verdict, bounding-box preview, vehicle stats, confidence, and the track IDs involved in the overlap.
4. **Training-ready pipeline** (`tools/extract_sequences.py` + `train.py`): converts labeled clips into fixed-length sequences for a CNN+LSTM classifier and reports precision/recall + confusion matrix. Trained models land in `models/trained/`.

## Detection Setup
1. Put the YOLO artifacts (`yolov3-tiny.cfg`, `yolov3-tiny.weights`, `coco.names`) inside `models/`.
2. Start the web server:
   ```
   pip install -r requirements.txt
   python app.py
   ```
3. Open the browser at `http://127.0.0.1:5000/`, upload a `.mp4`, and inspect the annotated preview plus stats.

## Dataset Organization
The training pipeline expects the following structure inside `Datasets/`:

```
Datasets/
  raw/
    accident/       # place labeled MP4 clips showing collisions
    no_accident/    # place clips without collisions
  sequences/
    accident/<video>__seq000/000.jpg ...        # generated image sequences
    no_accident/<video>__seq000/000.jpg ...
  index.csv         # populated by tools/extract_sequences.py
```

`tools/extract_sequences.py` downsamples each video to ~10 FPS, crops it into `seq_len` windows, resizes to `112x112`, and writes an `index.csv` that `train.py` consumes.

## Training Pipeline
1. Install training dependencies:
   ```
   pip install -r requirements-train.txt
   # install torch/torchvision manually (see https://pytorch.org/get-started/locally/)
   ```
2. Run the extractor:
   ```
   python tools/extract_sequences.py --datasets_dir Datasets/ --seq_len 16
   ```
3. Train the CNN+LSTM classifier:
   ```
   python train.py --epochs 10 --batch_size 8 --seq_len 16
   ```

Training evaluates on validation/test splits, prints precision/recall, saves the confusion matrix, and writes the best model into `models/trained/cnn_lstm_accident_best.pt`.

## Notes for Developers
- The detection API is intentionally stateless: uploaded clips stay in `static/uploads/` and can be inspected later.
- The front-end consumes the preview image returned as Base64, so no extra files are created server-side.
- The training stack uses vanilla `torch.nn` modules — no complex architectures — so you can extend the dataset or swap in other classifiers later.