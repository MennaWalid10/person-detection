
import cv2
from pathlib import Path

VIDEO_DIR = Path("dataset/vid")
OUTPUT_DIR = Path("dataset/images/raw")

FRAME_INTERVAL = 1  # one frame per second

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = list(VIDEO_DIR.glob("*.mp4"))

print(f"Found {len(videos)} videos")

for video in videos:
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"Cannot open: {video.name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps * FRAME_INTERVAL))

    folder = OUTPUT_DIR / video.stem
    folder.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % interval == 0:
            filename = folder / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(filename), frame)
            saved += 1

        frame_number += 1

    cap.release()
    print(f"{video.name}: {saved} frames saved")

print("Extraction completed!")
