import cv2
import numpy as np
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parent.parent / "data" / "test_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
output_path = VIDEO_DIR / "sample_walkthrough.mp4"

width, height = 640, 480
fps = 25
total_frames = 25 * 18  # 18 seconds test video

# Use mp4v fourcc
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

print(f"Generating synthetic testcase video at {output_path}...")

for f in range(total_frames):
    t = f / fps
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (32, 34, 40)

    # Architectural Hallway Grid
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (44, 46, 54), 1)
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (44, 46, 54), 1)

    # Doorway / Restricted Vault Area outline in the background
    cv2.rectangle(img, (200, 100), (440, 400), (55, 60, 70), -1)
    cv2.rectangle(img, (240, 140), (400, 390), (40, 45, 55), -1)
    cv2.putText(img, "SECURE VAULT DOOR", (250, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 140, 160), 1)

    # Scenario 1: Person walking into Restricted Zone (Vault) from t = 1.0s to 8.0s
    if 1.0 <= t <= 8.0:
        # Starts at x=50, walks to x=320 (inside restricted vault at x in [120..520], y in [100..400])
        progress = min(1.0, (t - 1.0) / 4.0)
        px = int(50 + progress * 270)
        py = int(320 - progress * 20)
        # Person torso & head
        cv2.circle(img, (px, py - 45), 18, (190, 170, 140), -1)
        cv2.rectangle(img, (px - 22, py - 27), (px + 22, py + 45), (140, 70, 40), -1)
        # Legs
        cv2.line(img, (px - 10, py + 45), (px - 10, py + 80), (60, 60, 80), 6)
        cv2.line(img, (px + 10, py + 45), (px + 10, py + 80), (60, 60, 80), 6)
        cv2.putText(img, "TEST TARGET A", (px - 35, py - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Scenario 2: Person entering and standing in Loiter zone from t = 7.0s to 18.0s
    if t >= 7.0:
        # Person stops at (160, 260) and loiters
        lx = 160
        ly = 260
        cv2.circle(img, (lx, ly - 45), 18, (170, 190, 140), -1)
        cv2.rectangle(img, (lx - 22, ly - 27), (lx + 22, ly + 45), (40, 100, 140), -1)
        cv2.line(img, (lx - 10, ly + 45), (lx - 10, ly + 80), (60, 60, 80), 6)
        cv2.line(img, (lx + 10, ly + 45), (lx + 10, ly + 80), (60, 60, 80), 6)
        cv2.putText(img, "TEST TARGET B", (lx - 35, ly - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Backpack placed at (120, 310)
        cv2.rectangle(img, (105, 305), (135, 335), (30, 80, 160), -1)
        cv2.putText(img, "BAG", (105, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 255), 1)

    # Timecode Stamp
    cv2.putText(img, f"REC [TC {t:05.2f}s | F {f:04d}]", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out.write(img)

out.release()
print(f"Sample testcase video generated successfully: {output_path} ({total_frames} frames, {output_path.stat().st_size} bytes)")
