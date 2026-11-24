
import cv2
import csv
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ==================================
# Configuration
# ==================================

MODEL = "yolo11n.onnx"
VIDEO_DIR = Path("dataset/vid")

OUTPUT_DIR = Path("results/video_evaluation/coco_nms")
SAMPLES_DIR = OUTPUT_DIR / "samples"

IMAGE_SIZE = 320
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 10
SAMPLE_INTERVAL = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ==================================
# Load Model
# ==================================

print("Loading COCO YOLO11n ONNX...")
model = YOLO(MODEL)

video_files = sorted(VIDEO_DIR.glob("*.mp4"))

if not video_files:
    raise FileNotFoundError("No MP4 videos found.")

all_results = []

# ==================================
# Evaluate each video
# ==================================

for video_path in video_files:

    print("\n" + "=" * 60)
    print("Evaluating:", video_path.name)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open video.")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed = 0
    frames_with_person = 0
    total_detections = 0

    confidences = []
    inference_times = []
    total_times = []

    video_samples = SAMPLES_DIR / video_path.stem
    video_samples.mkdir(parents=True, exist_ok=True)

    while True:

        frame_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            break

        inference_start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            classes=[0],
            verbose=False
        )[0]

        inference_end = time.perf_counter()

        boxes = result.boxes
        count = len(boxes)

        processed += 1
        total_detections += count

        if count > 0:
            frames_with_person += 1

            for box in boxes:
                confidences.append(float(box.conf[0].item()))

        inference_times.append(
            (inference_end - inference_start) * 1000
        )

        frame_end = time.perf_counter()
        total_times.append(frame_end - frame_start)

        if (processed - 1) % SAMPLE_INTERVAL == 0:
            annotated = result.plot()

            cv2.imwrite(
                str(video_samples / f"frame_{processed:05d}.jpg"),
                annotated
            )

        if processed % 100 == 0:
            print(f"Processed {processed}/{total_frames}")

    cap.release()

    if processed == 0:
        continue

    row = {
        "video": video_path.name,
        "total_frames": total_frames,
        "processed_frames": processed,
        "frames_with_person": frames_with_person,
        "frames_without_detection": processed - frames_with_person,
        "detection_rate_percent": round(
            frames_with_person / processed * 100, 2
        ),
        "total_person_detections": total_detections,
        "average_confidence": round(
            float(np.mean(confidences)) if confidences else 0, 4
        ),
        "average_inference_ms": round(
            float(np.mean(inference_times)), 2
        ),
        "p95_inference_ms": round(
            float(np.percentile(inference_times, 95)), 2
        ),
        "end_to_end_fps": round(
            processed / sum(total_times), 2
        )
    }

    all_results.append(row)

    print("\nVideo Results:")
    for key, value in row.items():
        print(f"{key}: {value}")

# ==================================
# Save results
# ==================================

csv_path = OUTPUT_DIR / "video_metrics.csv"

if all_results:

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nEvaluation completed!")
    print("CSV:", csv_path)
    print("Samples:", SAMPLES_DIR)

else:
    print("No results generated.")
