import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from pathlib import Path

VIDEO_DIR = Path("dataset/vid")
MODEL_DIR = Path("models/efficientdet_lite0")
BASE_OUTPUT = Path("results/efficientdet_lite0")

CONFIDENCE = 0.30
SAMPLE_INTERVAL = 30

MODELS = {
    "FP32": MODEL_DIR / "efficientdet_lite0_float32.tflite",
    "INT8": MODEL_DIR / "efficientdet_lite0_int8.tflite"
}

mp_vision = mp.tasks.vision
all_results = []

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 60)
    print("MODEL:", model_name)
    print("=" * 60)

    output_dir = BASE_OUTPUT / model_name
    missed_dir = output_dir / "missed_frames"
    samples_dir = output_dir / "samples"

    missed_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    options = mp_vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        score_threshold=CONFIDENCE,
        category_allowlist=["person"],
        max_results=10
    )

    with mp_vision.ObjectDetector.create_from_options(options) as detector:

        for video_path in sorted(VIDEO_DIR.glob("*.mp4")):

            print("\nEvaluating:", video_path.name)

            cap = cv2.VideoCapture(str(video_path))

            total_frames = 0
            frames_with_detection = 0
            total_detections = 0
            confidences = []
            inference_times = []

            video_missed_dir = missed_dir / video_path.stem
            video_missed_dir.mkdir(parents=True, exist_ok=True)

            video_samples_dir = samples_dir / video_path.stem
            video_samples_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.perf_counter()

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                total_frames += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                t0 = time.perf_counter()
                result = detector.detect(mp_image)
                inference_ms = (time.perf_counter() - t0) * 1000

                inference_times.append(inference_ms)

                detections = result.detections

                if detections:
                    frames_with_detection += 1
                    total_detections += len(detections)

                    for detection in detections:
                        confidence = detection.categories[0].score
                        confidences.append(confidence)

                else:
                    cv2.imwrite(
                        str(video_missed_dir / f"frame_{total_frames:05d}.jpg"),
                        frame
                    )

                if total_frames % SAMPLE_INTERVAL == 0:
                    sample = frame.copy()

                    for detection in detections:
                        box = detection.bounding_box

                        x1 = max(0, box.origin_x)
                        y1 = max(0, box.origin_y)
                        x2 = min(frame.shape[1], box.origin_x + box.width)
                        y2 = min(frame.shape[0], box.origin_y + box.height)

                        score = detection.categories[0].score

                        cv2.rectangle(
                            sample, (x1, y1), (x2, y2),
                            (0, 255, 0), 2
                        )

                        cv2.putText(
                            sample,
                            f"person {score:.2f}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2
                        )

                    cv2.imwrite(
                        str(video_samples_dir / f"frame_{total_frames:05d}.jpg"),
                        sample
                    )

            cap.release()

            elapsed = time.perf_counter() - start_time

            avg_conf = (
                float(np.mean(confidences)) if confidences else 0
            )

            avg_inference = (
                float(np.mean(inference_times)) if inference_times else 0
            )

            p95_inference = (
                float(np.percentile(inference_times, 95))
                if inference_times else 0
            )

            coverage = (
                frames_with_detection / total_frames * 100
                if total_frames else 0
            )

            e2e_fps = total_frames / elapsed if elapsed else 0

            row = {
                "model": model_name,
                "video": video_path.name,
                "total_frames": total_frames,
                "frames_with_detection": frames_with_detection,
                "frames_without_detection": total_frames - frames_with_detection,
                "detection_coverage_percent": round(coverage, 2),
                "total_person_detections": total_detections,
                "average_confidence": round(avg_conf, 4),
                "average_inference_ms": round(avg_inference, 2),
                "p95_inference_ms": round(p95_inference, 2),
                "end_to_end_fps": round(e2e_fps, 2)
            }

            all_results.append(row)

            print(row)

    df = pd.DataFrame(
        [r for r in all_results if r["model"] == model_name]
    )

    df.to_csv(output_dir / "video_metrics.csv", index=False)

    print("\nSaved:", output_dir / "video_metrics.csv")

final_df = pd.DataFrame(all_results)
final_df.to_csv(BASE_OUTPUT / "comparison_results.csv", index=False)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for model_name in MODELS:
    subset = final_df[final_df["model"] == model_name]

    total = subset["total_frames"].sum()
    detected = subset["frames_with_detection"].sum()

    print(
        model_name,
        "| Frames:", total,
        "| Detected:", detected,
        "| Coverage:", round(detected / total * 100, 2), "%"
    )

print("\nComparison CSV:", BASE_OUTPUT / "comparison_results.csv")
