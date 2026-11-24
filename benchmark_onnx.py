
import cv2
import time
import csv
import psutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODEL_PATH = Path(
    "runs/detect/runs/yolo11n_video_split_320/weights/best_ncnn_model"
)

IMAGE_DIR = Path("dataset/video_split/test/images")
OUTPUT_CSV = Path("results/csv/ncnn_benchmark.csv")

IMG_SIZE = 320
NUM_FRAMES = 300
WARMUP = 30


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    images = sorted(IMAGE_DIR.glob("*"))

    if not images:
        raise RuntimeError("No test images found.")

    model = YOLO(str(MODEL_PATH), task="detect")

    frames = []

    for path in images:
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)

    if not frames:
        raise RuntimeError("Could not read test images.")

    print("Model: YOLO11n NCNN")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Loaded {len(frames)} test images.")
    print("Starting warm-up...")

    for i in range(WARMUP):
        model.predict(
            source=frames[i % len(frames)],
            imgsz=IMG_SIZE,
            device="cpu",
            verbose=False
        )

    print("Warm-up completed.")
    print("Benchmark started...")

    latencies = []
    process = psutil.Process()

    cpu_samples = []
    ram_samples = []

    psutil.cpu_percent(interval=None)

    start_total = time.perf_counter()

    for i in range(NUM_FRAMES):

        frame = frames[i % len(frames)]

        start = time.perf_counter()

        model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            device="cpu",
            verbose=False
        )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(
            process.memory_info().rss / (1024 ** 2)
        )

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{NUM_FRAMES}")

    total_time = time.perf_counter() - start_total

    results = {
        "Model": "YOLO11n",
        "Format": "NCNN",
        "Image Size": IMG_SIZE,
        "Frames": NUM_FRAMES,
        "Average Latency ms": round(float(np.mean(latencies)), 2),
        "P95 Latency ms": round(float(np.percentile(latencies, 95)), 2),
        "Inference FPS": round(NUM_FRAMES / total_time, 2),
        "Average CPU %": round(float(np.mean(cpu_samples)), 2),
        "Average RAM MB": round(float(np.mean(ram_samples)), 2),
        "Model Size MB": round(
            sum(p.stat().st_size for p in MODEL_PATH.iterdir() if p.is_file())
            / (1024 ** 2), 2
        ),
    }

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists()

    with OUTPUT_CSV.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(results)

    print("\n========== BENCHMARK RESULTS ==========")

    for key, value in results.items():
        print(f"{key}: {value}")

    print("=======================================")
    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
