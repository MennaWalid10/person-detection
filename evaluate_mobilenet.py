import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from pathlib import Path

VIDEO_DIR = Path("dataset/vid")
MODEL_DIR = Path("models/mobilenet_ssd")
OUTPUT_DIR = Path("results/mobilenet_ssd")

CONFIDENCE = 0.30
SAMPLE_INTERVAL = 30

MODELS = {
    "FP32": MODEL_DIR / "ssd_mobilenet_v2_float32.tflite",
    "FP16": MODEL_DIR / "ssd_mobilenet_v2_float16.tflite"
}

all_results = []

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)

    model_output = OUTPUT_DIR / model_name
    missed_root = model_output / "missed_frames"
    samples_root = model_output / "samples"

    missed_root.mkdir(parents=True, exist_ok=True)
    samples_root.mkdir(parents=True, exist_ok=True)

    options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        score_threshold=CONFIDENCE,
        category_allowlist=["person"],
        max_results=10
    )

    with mp.tasks.vision.ObjectDetector.create_from_options(options) as detector:

        for video_path in sorted(VIDEO_DIR.glob("*.mp4")):

            print("\nEvaluating:", video_path.name)

            cap = cv2.VideoCapture(str(video_path))

            total = detected = total_detections = 0
            confidences = []
            inference_times = []

            missed_dir = missed_root / video_path.stem
            sample_dir = samples_root / video_path.stem

            missed_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)

            start = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )

                t0 = time.perf_counter()
                result = detector.detect(image)
                inference_ms = (time.perf_counter() - t0) * 1000

                inference_times.append(inference_ms)
                detections = result.detections

                if detections:
                    detected += 1
                    total_detections += len(detections)

                    for d in detections:
                        confidences.append(d.categories[0].score)
                else:
                    cv2.imwrite(
                        str(missed_dir / f"frame_{total:05d}.jpg"),
                        frame
                    )

                if total % SAMPLE_INTERVAL == 0:
                    sample = frame.copy()

                    for d in detections:
                        b = d.bounding_box
                        x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
                        x2 = min(frame.shape[1], b.origin_x + b.width)
                        y2 = min(frame.shape[0], b.origin_y + b.height)

                        score = d.categories[0].score

                        cv2.rectangle(sample, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(
                            sample, f"person {score:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
                        )

                    cv2.imwrite(
                        str(sample_dir / f"frame_{total:05d}.jpg"),
                        sample
                    )

            cap.release()

            elapsed = time.perf_counter() - start

            row = {
                "model": model_name,
                "video": video_path.name,
                "total_frames": total,
                "frames_with_detection": detected,
                "frames_without_detection": total - detected,
                "detection_coverage_percent": round(detected / total * 100, 2) if total else 0,
                "total_person_detections": total_detections,
                "average_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
                "average_inference_ms": round(float(np.mean(inference_times)), 2) if inference_times else 0,
                "p95_inference_ms": round(float(np.percentile(inference_times, 95)), 2) if inference_times else 0,
                "end_to_end_fps": round(total / elapsed, 2) if elapsed else 0
            }

            all_results.append(row)
            print(row)

    pd.DataFrame(
        [r for r in all_results if r["model"] == model_name]
    ).to_csv(model_output / "video_metrics.csv", index=False)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)

print("\nFINAL SUMMARY")
for name in MODELS:
    subset = df[df["model"] == name]
    total = subset["total_frames"].sum()
    detected = subset["frames_with_detection"].sum()
    print(
        name, "| Frames:", total,
        "| Detected:", detected,
        "| Coverage:", round(detected / total * 100, 2), "%"
    )

print("\nCSV:", OUTPUT_DIR / "comparison_results.csv")
