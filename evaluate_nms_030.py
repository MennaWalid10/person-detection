
import cv2
import csv
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ================= CONFIGURATION =================

MODEL = "runs/detect/runs/detect/yolo11n_partial_finetune/weights/best.pt"

VIDEO_DIR = Path("dataset/vid")

OUTPUT_DIR = Path("results/video_evaluation/partial_finetuned_conf030_iou030")
SAMPLES_DIR = OUTPUT_DIR / "samples"
MISSED_DIR = OUTPUT_DIR / "missed_frames"

IMAGE_SIZE = 320
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.30
MAX_DETECTIONS = 10
SAMPLE_INTERVAL = 30

# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
MISSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Fine-tuned YOLO11n ONNX...")
model = YOLO(MODEL)

video_files = sorted(VIDEO_DIR.glob("*.mp4"))

if not video_files:
    raise FileNotFoundError(f"No MP4 videos found in {VIDEO_DIR}")

all_results = []

for video_path in video_files:

    print("\n" + "=" * 60)
    print(f"Evaluating: {video_path.name}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path.name}")
        continue

    total_frames = 0
    frames_with_person = 0
    frames_without_detection = 0
    total_detections = 0

    confidences = []
    inference_times = []

    video_samples = SAMPLES_DIR / video_path.stem
    video_missed = MISSED_DIR / video_path.stem

    video_samples.mkdir(parents=True, exist_ok=True)
    video_missed.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

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

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000

        inference_times.append(inference_ms)

        boxes = result.boxes
        detection_count = len(boxes)

        if detection_count > 0:
            frames_with_person += 1
            total_detections += detection_count

            for box in boxes:
                confidences.append(float(box.conf[0]))

        else:
            frames_without_detection += 1

            cv2.imwrite(
                str(video_missed / f"frame_{total_frames:05d}.jpg"),
                frame
            )

        # Save one annotated sample every 30 frames
        if total_frames % SAMPLE_INTERVAL == 0:
            annotated = result.plot()

            cv2.imwrite(
                str(video_samples / f"frame_{total_frames:05d}.jpg"),
                annotated
            )

    cap.release()

    elapsed = time.perf_counter() - start_time

    detection_rate = (
        frames_with_person / total_frames * 100
        if total_frames else 0
    )

    avg_confidence = (
        np.mean(confidences) if confidences else 0
    )

    avg_inference = (
        np.mean(inference_times) if inference_times else 0
    )

    p95_inference = (
        np.percentile(inference_times, 95)
        if inference_times else 0
    )

    end_to_end_fps = (
        total_frames / elapsed if elapsed > 0 else 0
    )

    row = {
        "video": video_path.name,
        "total_frames": total_frames,
        "frames_with_person": frames_with_person,
        "frames_without_detection": frames_without_detection,
        "detection_rate_percent": round(detection_rate, 2),
        "total_person_detections": total_detections,
        "average_confidence": round(avg_confidence, 4),
        "average_inference_ms": round(avg_inference, 2),
        "p95_inference_ms": round(p95_inference, 2),
        "end_to_end_fps": round(end_to_end_fps, 2)
    }

    all_results.append(row)

    print(f"Total frames: {total_frames}")
    print(f"Frames with detection: {frames_with_person}")
    print(f"Frames without detection: {frames_without_detection}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Total detections: {total_detections}")
    print(f"Average confidence: {avg_confidence:.4f}")
    print(f"Average inference: {avg_inference:.2f} ms")
    print(f"P95 inference: {p95_inference:.2f} ms")
    print(f"End-to-end FPS: {end_to_end_fps:.2f}")

# Save CSV results
csv_path = OUTPUT_DIR / "video_metrics.csv"

if all_results:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    total = sum(r["total_frames"] for r in all_results)
    detected = sum(r["frames_with_person"] for r in all_results)
    missed = sum(r["frames_without_detection"] for r in all_results)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(all_results)}")
    print(f"Total frames: {total}")
    print(f"Frames with detection: {detected}")
    print(f"Frames without detection: {missed}")
    print(f"Overall detection coverage: {detected / total * 100:.2f}%")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Missed frames: {MISSED_DIR}")
    print(f"Sample frames: {SAMPLES_DIR}")
else:
    print("No videos were evaluated.")
