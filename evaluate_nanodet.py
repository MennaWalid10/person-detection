
import cv2
import numpy as np
import onnxruntime as ort
import glob
import os
import time
import csv

MODEL = "models/nanodet/nanodet_plus_m_320.onnx"
VIDEO_FOLDER = "dataset/vid"
OUTPUT = "results/nanodet"
IMG_SIZE = 320
CONF = 0.30
IOU = 0.50

STRIDES = [8, 16, 32, 64]
REG_MAX = 7

MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def make_anchors():
    levels = []
    for stride in STRIDES:
        size = IMG_SIZE // stride
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        anchors = np.column_stack((
            x.flatten() * stride + 0.5 * (stride - 1),
            y.flatten() * stride + 0.5 * (stride - 1)
        ))
        levels.append(anchors)
    return levels


ANCHORS = make_anchors()


def decode(output):
    output = output[0]
    all_boxes, all_scores = [], []
    offset = 0

    for stride, anchors in zip(STRIDES, ANCHORS):
        count = len(anchors)
        pred = output[offset:offset + count]
        offset += count

        scores = pred[:, :80]
        person_scores = scores[:, 0]

        reg = pred[:, 80:].reshape(-1, 4, REG_MAX + 1)
        reg = reg - reg.max(axis=2, keepdims=True)
        prob = np.exp(reg)
        prob /= prob.sum(axis=2, keepdims=True)

        distances = np.sum(
            prob * np.arange(REG_MAX + 1, dtype=np.float32),
            axis=2
        ) * stride

        boxes = np.column_stack((
            anchors[:, 0] - distances[:, 0],
            anchors[:, 1] - distances[:, 1],
            anchors[:, 0] + distances[:, 2],
            anchors[:, 1] + distances[:, 3]
        ))

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, IMG_SIZE)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, IMG_SIZE)

        all_boxes.append(boxes)
        all_scores.append(person_scores)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)

    valid = scores >= CONF
    boxes, scores = boxes[valid], scores[valid]

    if len(boxes) == 0:
        return [], []

    nms_boxes = [
        [float(b[0]), float(b[1]),
         float(b[2] - b[0]), float(b[3] - b[1])]
        for b in boxes
    ]

    indices = cv2.dnn.NMSBoxes(
        nms_boxes, scores.tolist(), CONF, IOU
    )

    if len(indices) == 0:
        return [], []

    indices = np.array(indices).flatten()
    return boxes[indices], scores[indices]


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    session = ort.InferenceSession(
        MODEL, providers=["CPUExecutionProvider"]
    )

    videos = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")))
    if not videos:
        print("No videos found!")
        return

    rows = []

    for video in videos:
        name = os.path.basename(video)
        print("\nEvaluating:", name)

        cap = cv2.VideoCapture(video)
        total = 0
        detected = 0
        confidences = []
        latencies = []
        missed_dir = os.path.join(
            OUTPUT, "missed_frames",
            os.path.splitext(name)[0]
        )
        os.makedirs(missed_dir, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total += 1
            h, w = frame.shape[:2]

            resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            inp = (resized.astype(np.float32) - MEAN) / STD
            inp = np.transpose(inp, (2, 0, 1))[None, ...]

            start = time.perf_counter()
            output = session.run(None, {"data": inp})[0]
            boxes, scores = decode(output)
            latencies.append((time.perf_counter() - start) * 1000)

            if len(boxes):
                detected += 1
                confidences.extend(scores.tolist())
            else:
                cv2.imwrite(
                    os.path.join(missed_dir, f"frame_{total:06d}.jpg"),
                    frame
                )

        cap.release()

        coverage = detected / total * 100 if total else 0
        avg_ms = float(np.mean(latencies)) if latencies else 0
        p95_ms = float(np.percentile(latencies, 95)) if latencies else 0
        fps = 1000 / avg_ms if avg_ms else 0
        avg_conf = float(np.mean(confidences)) if confidences else 0

        row = {
            "model": "NanoDet-Plus-m-320",
            "video": name,
            "total_frames": total,
            "frames_with_person": detected,
            "frames_without_detection": total - detected,
            "detection_coverage_percent": round(coverage, 2),
            "average_confidence": round(avg_conf, 4),
            "average_inference_ms": round(avg_ms, 2),
            "p95_inference_ms": round(p95_ms, 2),
            "end_to_end_fps": round(fps, 2)
        }

        rows.append(row)
        print(row)

    csv_path = os.path.join(OUTPUT, "comparison_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_frames = sum(r["total_frames"] for r in rows)
    total_detected = sum(r["frames_with_person"] for r in rows)

    print("\nFINAL SUMMARY")
    print(f"Frames: {total_frames} | Detected: {total_detected}")
    print(f"Coverage: {total_detected / total_frames * 100:.2f}%")
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
