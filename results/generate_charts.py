from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results")
OUT = ROOT / "charts"
OUT.mkdir(parents=True, exist_ok=True)

datasets = [
    ("YOLO11n COCO ONNX", "video_evaluation/coco_nms_030/video_metrics.csv"),
    ("YOLO11n Fine-tuned ONNX", "video_evaluation/finetuned_conf030_iou030/video_metrics.csv"),
    ("EfficientDet-Lite0 FP32", "efficientdet_lite0/FP32/video_metrics.csv"),
    ("EfficientDet-Lite0 INT8", "efficientdet_lite0/INT8/video_metrics.csv"),
    ("MobileNet-SSD FP32", "mobilenet_ssd/FP32/video_metrics.csv"),
    ("MobileNet-SSD FP16", "mobilenet_ssd/FP16/video_metrics.csv"),
    ("NanoDet-Plus", "nanodet/comparison_results.csv"),
]

results = []

for model, relative_path in datasets:
    path = ROOT / relative_path
    if not path.exists():
        print(f"SKIPPED missing: {path}")
        continue

    df = pd.read_csv(path)

    detected_col = next(
        (c for c in ["frames_with_person", "frames_with_detection"]
         if c in df.columns), None
    )

    if detected_col is None:
        print(f"SKIPPED: no detection column in {path}")
        continue

    for col in ["total_frames", detected_col,
                "average_inference_ms", "end_to_end_fps"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_frames = df["total_frames"].sum()
    detected_frames = df[detected_col].sum()

    valid = df.dropna(
        subset=["total_frames", "average_inference_ms", "end_to_end_fps"]
    )
    valid = valid[valid["end_to_end_fps"] > 0]

    avg_latency = (
        (valid["average_inference_ms"] * valid["total_frames"]).sum()
        / valid["total_frames"].sum()
    )

    total_elapsed = (
        valid["total_frames"] / valid["end_to_end_fps"]
    ).sum()

    aggregate_fps = valid["total_frames"].sum() / total_elapsed

    results.append({
        "Model": model,
        "Total_Frames": int(total_frames),
        "Detected_Frames": int(detected_frames),
        "Coverage_Percent": 100 * detected_frames / total_frames,
        "Average_Inference_ms": avg_latency,
        "Aggregate_End_to_End_FPS": aggregate_fps,
    })

summary = pd.DataFrame(results)
summary.to_csv(OUT / "chart_summary.csv", index=False)

# Chart 1: Detection coverage vs end-to-end FPS
plt.figure(figsize=(10, 6))
plt.scatter(
    summary["Aggregate_End_to_End_FPS"],
    summary["Coverage_Percent"],
    s=90
)

for _, row in summary.iterrows():
    plt.annotate(
        row["Model"],
        (row["Aggregate_End_to_End_FPS"], row["Coverage_Percent"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("Aggregate End-to-End FPS")
plt.ylabel("Detection Coverage (%)")
plt.title("Detection Coverage vs End-to-End FPS")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "accuracy_vs_fps.png", dpi=300)
plt.close()

# Chart 2: Average inference latency
ordered = summary.sort_values("Average_Inference_ms")

plt.figure(figsize=(10, 6))
plt.barh(ordered["Model"], ordered["Average_Inference_ms"])
plt.xlabel("Average Inference Latency (ms)")
plt.ylabel("Model")
plt.title("Average Inference Latency Comparison")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "latency_comparison.png", dpi=300)
plt.close()

print("\nCHARTS GENERATED")
print(summary.round(2).to_string(index=False))
print(f"\nSaved in: {OUT.resolve()}")
