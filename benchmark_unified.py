
import cv2
import time
import csv
import psutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuration
MODELS = {
    "YOLO11n_ONNX": Path(
        "runs/detect/runs/yolo11n_video_split_320/weights/best.onnx"
    ),
    "YOLO11n_NCNN": Path(
        "runs/detect/runs/yolo11n_video_split_320/weights/best_ncnn_model"
    ),
}

IMAGE_DIR = Path("dataset/video_split/test/images")
OUTPUT_CSV = Path("results/csv/unified_benchmark.csv")

IMG_SIZE = 320
NUM_FRAMES = 300
WARMUP = 30


def benchmark_model(model_name, model_path, frames):

    print(f"\nLoading {model_name}...")
    model = YOLO(str(model_path), task="detect")

    print("Starting warm-up...")
    for i in range(WARMUP):
        model.predict(
            source=frames[i % len(frames)],
            imgsz=IMG_SIZE,
            device="cpu",
            verbose=False
        )

    print("Warm-up completed.")

    latencies = []
    cpu_samples = []
    ram_samples = []

    process = psutil.Process()
    psutil.cpu_percent(interval=None)

    print("Benchmark started...")
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

        latencies.append((time.perf_counter() - start) * 1000)

        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(process.memory_info().rss / (1024 ** 2))

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{NUM_FRAMES}")

    total_time = time.perf_counter() - start_total

    if model_path.is_dir():
        model_size = sum(
            p.stat().st_size for p in model_path.rglob("*") if p.is_file()
        ) / (1024 ** 2)
    else:
        model_size = model_path.stat().st_size / (1024 ** 2)

    return {
        "Model": "YOLO11n",
        "Format": model_name.split("_")[-1],
        "Image Size": IMG_SIZE,
        "Frames": NUM_FRAMES,
        "Average Latency ms": round(float(np.mean(latencies)), 2),
        "P95 Latency ms": round(float(np.percentile(latencies, 95)), 2),
        "Inference FPS": round(NUM_FRAMES / total_time, 2),
        "Average CPU %": round(float(np.mean(cpu_samples)), 2),
        "Average RAM MB": round(float(np.mean(ram_samples)), 2),
        "Model Size MB": round(model_size, 2),
    }


def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    frames = []
    for path in sorted(IMAGE_DIR.glob("*")):
        image = cv2.imread(str(path))
        if image is not None:
            frames.append(image)

    if not frames:
        raise RuntimeError("No readable test images found.")

    print(f"Loaded {len(frames)} test images.")

    all_results = []

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        result = benchmark_model(model_name, model_path, frames)
        all_results.append(result)

        print("\n========== RESULTS ==========")
        for key, value in result.items():
            print(f"{key}: {value}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nUnified results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
