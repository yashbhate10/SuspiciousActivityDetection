"""
Enterprise Dataset Builder v3 for UCF-Crime Classification.

Root causes of v2's 34% accuracy (FIXED here):
  1. Normal class had only 100 unique source frames (80% were augmented copies)
     → Now: extract 40 frames per video from ALL crime categories + uniformly
       sample across entire video timeline for maximum scene diversity
  2. Crime frames were nearly identical (consecutive frames from same video)
     → Now: extract frames at wider intervals, skip near-duplicates using
       pixel difference check
  3. Augmentation was too heavy relative to real data (4:1 aug:real ratio)
     → Now: only augment to fill gaps, max 1:1 ratio
  4. Normal frames came from same videos as crimes (same scene/lighting)
     → Now: extract Normal frames from DIFFERENT videos than used for crimes
       (cross-video split), and use more frames per video with wider spacing

Produces: Normal, Abuse, Arrest, Arson, Assault (5 classes)
"""

import os
import sys
import random
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "Anomaly-Videos-Part-1 (1)" / "Anomaly-Videos-Part-1"
TARGET_DATASET_DIR = PROJECT_ROOT / "data" / "anomaly_cls_v3"

CRIME_CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault"]
ALL_CATEGORIES = ["Normal"] + CRIME_CATEGORIES

# ── Frame Quality Helpers ────────────────────────────────────────────

def is_different_enough(frame: np.ndarray, prev_frame: np.ndarray, threshold: float = 12.0) -> bool:
    """Check if two frames are visually different enough (avoids near-duplicate frames)."""
    if prev_frame is None:
        return True
    diff = cv2.absdiff(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    )
    return float(np.mean(diff)) > threshold


def is_valid_frame(frame: np.ndarray) -> bool:
    """Reject black/blank/corrupt frames."""
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    # Reject near-black or near-uniform frames
    return mean_intensity > 15 and std_intensity > 10


def extract_frames_uniform(
    video_path: Path,
    max_frames: int,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
    deduplicate: bool = True
) -> list:
    """
    Extract up to max_frames frames uniformly spread between start_pct and end_pct
    of the video. Skips near-duplicate and invalid frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 30:
        cap.release()
        return []

    start_frame = int(total * start_pct)
    end_frame = int(total * end_pct) - 1
    span = end_frame - start_frame
    if span < max_frames:
        # Not enough frames in this range
        step = 1
    else:
        step = span // max_frames

    target_indices = list(range(start_frame, end_frame, max(1, step)))[:max_frames * 2]

    frames = []
    prev_frame = None
    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if not is_valid_frame(frame):
            continue
        if deduplicate and not is_different_enough(frame, prev_frame):
            continue
        frames.append(frame)
        prev_frame = frame
        if len(frames) >= max_frames:
            break

    cap.release()
    return frames


# ── Augmentation (conservative — only to fill class gaps) ───────────

def augment_batch(frames: list, target_count: int) -> list:
    """
    Apply LIGHT augmentation to reach target_count.
    Max 1:1 augmented:real ratio to prevent overfitting to augmented data.
    """
    if len(frames) >= target_count:
        return frames[:target_count]

    augmented = list(frames)  # start with originals
    max_aug = len(frames)  # at most as many augmented as originals (1:1)
    deficit = target_count - len(frames)
    to_generate = min(deficit, max_aug)

    source = list(frames)
    random.shuffle(source)

    generated = 0
    for frame in source:
        if generated >= to_generate:
            break
        # Pick a random augmentation
        aug_type = random.choice(["flip", "brightness", "contrast", "crop"])
        h, w = frame.shape[:2]

        if aug_type == "flip":
            aug = cv2.flip(frame, 1)
        elif aug_type == "brightness":
            delta = random.randint(-35, 35)
            aug = np.clip(frame.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        elif aug_type == "contrast":
            alpha = random.uniform(0.7, 1.3)
            aug = np.clip(frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        elif aug_type == "crop":
            # Random crop & resize back
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            x1 = random.randint(0, margin_x)
            y1 = random.randint(0, margin_y)
            x2 = w - random.randint(0, margin_x)
            y2 = h - random.randint(0, margin_y)
            aug = cv2.resize(frame[y1:y2, x1:x2], (w, h))
        else:
            aug = frame

        augmented.append(aug)
        generated += 1

    return augmented[:target_count]


# ── Main Dataset Builder ────────────────────────────────────────────

def build_dataset(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    target_dir: Path = TARGET_DATASET_DIR,
    crime_frames_per_video: int = 30,
    normal_frames_per_video: int = 25,
    train_ratio: float = 0.80,
    target_per_class: int = 800,
    seed: int = 42,
):
    print("=" * 65)
    print("  ENTERPRISE DATASET BUILDER v3")
    print("  Fixes: cross-video Normal split, deduplication, balanced aug")
    print("=" * 65)
    print(f"  Source : {source_dir}")
    print(f"  Target : {target_dir}")
    print(f"  Classes: {ALL_CATEGORIES}")
    print(f"  Frames per crime video: {crime_frames_per_video}")
    print(f"  Frames per normal video: {normal_frames_per_video}")
    print(f"  Target frames per class: {target_per_class}")
    print("=" * 65)

    if not source_dir.exists():
        alt = PROJECT_ROOT / "data" / "Anomaly-Videos-Part-1"
        if alt.exists():
            source_dir = alt
        else:
            raise FileNotFoundError(f"Source not found: {source_dir}")

    random.seed(seed)
    np.random.seed(seed)

    # Clean and recreate
    if target_dir.exists():
        print(f"\n  Removing old dataset at {target_dir}...")
        shutil.rmtree(target_dir)

    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    for cat in ALL_CATEGORIES:
        (train_dir / cat).mkdir(parents=True, exist_ok=True)
        (val_dir / cat).mkdir(parents=True, exist_ok=True)

    stats = {"train": {c: 0 for c in ALL_CATEGORIES},
             "val": {c: 0 for c in ALL_CATEGORIES}}

    # ── Collect all videos per category ──────────────────────────────
    category_videos = {}
    for cat in CRIME_CATEGORIES:
        cat_dir = source_dir / cat
        if not cat_dir.exists():
            print(f"\n  WARNING: {cat_dir} not found — skipping")
            category_videos[cat] = []
            continue
        vids = sorted(list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.avi")))
        random.shuffle(vids)
        category_videos[cat] = vids

    # ── Split videos into train/val PER CATEGORY ─────────────────────
    # For Normal: use first 30% of EACH category's train videos for Normal
    # For Crime: use remaining 70% of each category's train videos
    # This ensures Normal comes from different video instances than Crime frames

    for cat in CRIME_CATEGORIES:
        vids = category_videos[cat]
        if not vids:
            continue

        split = int(len(vids) * train_ratio)
        train_vids = vids[:split]
        val_vids = vids[split:]

        # Further split train_vids: first 30% → Normal source, last 70% → Crime source
        normal_split = int(len(train_vids) * 0.30)
        normal_source_vids = train_vids[:normal_split]
        crime_source_vids = train_vids[normal_split:]

        print(f"\n  [{cat}] {len(vids)} videos total")
        print(f"    Train: {len(normal_source_vids)} → Normal, {len(crime_source_vids)} → Crime")
        print(f"    Val:   {len(val_vids)} → Crime + Normal")

        # ── Extract CRIME frames (middle 60% of each video) ──────────
        for phase, vid_list, dest in [("train", crime_source_vids, train_dir),
                                       ("val", val_vids, val_dir)]:
            crime_frames = []
            for v in vid_list:
                frames = extract_frames_uniform(
                    v, max_frames=crime_frames_per_video,
                    start_pct=0.15, end_pct=0.85
                )
                crime_frames.extend(frames)

            # Augment if needed, save
            if phase == "train":
                crime_frames = augment_batch(crime_frames, target_per_class)
            
            for i, frame in enumerate(crime_frames):
                fname = f"{cat}_{phase}_{i:06d}.jpg"
                cv2.imwrite(str(dest / cat / fname), frame)
                stats[phase][cat] += 1

            print(f"    {phase} Crime [{cat}]: {stats[phase][cat]} frames")

        # ── Extract NORMAL frames from Normal-source videos ──────────
        # Uses the ENTIRE video timeline (these videos' crime parts are NOT
        # being used for crime training, so there's no data leakage)
        for v in normal_source_vids:
            frames = extract_frames_uniform(
                v, max_frames=normal_frames_per_video,
                start_pct=0.0, end_pct=1.0
            )
            for i, frame in enumerate(frames):
                count = stats["train"]["Normal"]
                fname = f"normal_{cat}_{v.stem}_{count:06d}.jpg"
                cv2.imwrite(str(train_dir / "Normal" / fname), frame)
                stats["train"]["Normal"] += 1

        # Normal val frames from val videos (edges only — safe)
        for v in val_vids:
            for region in [(0.0, 0.08), (0.92, 1.0)]:
                frames = extract_frames_uniform(
                    v, max_frames=5,
                    start_pct=region[0], end_pct=region[1]
                )
                for frame in frames:
                    count = stats["val"]["Normal"]
                    fname = f"normal_val_{cat}_{v.stem}_{count:06d}.jpg"
                    cv2.imwrite(str(val_dir / "Normal" / fname), frame)
                    stats["val"]["Normal"] += 1

    # ── Balance Normal class with augmentation ───────────────────────
    normal_train_dir = train_dir / "Normal"
    existing = list(normal_train_dir.glob("*.jpg"))
    print(f"\n  [Normal] {len(existing)} unique frames extracted")

    if len(existing) < target_per_class:
        print(f"  Augmenting Normal from {len(existing)} → {target_per_class}...")
        # Load existing frames
        existing_frames = []
        for f in existing:
            img = cv2.imread(str(f))
            if img is not None:
                existing_frames.append(img)
        augmented = augment_batch(existing_frames, target_per_class)
        # Save only the new augmented frames
        for i, frame in enumerate(augmented[len(existing_frames):]):
            fname = f"normal_aug_{i:06d}.jpg"
            cv2.imwrite(str(normal_train_dir / fname), frame)
            stats["train"]["Normal"] += 1
    elif len(existing) > target_per_class:
        # Trim excess
        random.shuffle(existing)
        for f in existing[target_per_class:]:
            f.unlink()
        stats["train"]["Normal"] = target_per_class

    # Final count
    stats["train"]["Normal"] = len(list(normal_train_dir.glob("*.jpg")))
    stats["val"]["Normal"] = len(list((val_dir / "Normal").glob("*.jpg")))

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DATASET v3 BUILD COMPLETE")
    print("=" * 65)
    for phase in ["train", "val"]:
        total = sum(stats[phase].values())
        print(f"\n  {phase.upper()} ({total} total):")
        for cat in ALL_CATEGORIES:
            count = stats[phase][cat]
            print(f"    • {cat:<10} {count:>5} frames")

    print(f"\n  Dataset ready at: {target_dir}")
    print("  Next: python ai-engine/train_anomaly.py --data data/anomaly_cls_v3 --epochs 50 --imgsz 224 --model yolo11m-cls.pt")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Dataset Builder v3")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--target", default=str(TARGET_DATASET_DIR))
    parser.add_argument("--crime-frames", type=int, default=30)
    parser.add_argument("--normal-frames", type=int, default=25)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--target-per-class", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_dataset(
        source_dir=Path(args.source),
        target_dir=Path(args.target),
        crime_frames_per_video=args.crime_frames,
        normal_frames_per_video=args.normal_frames,
        train_ratio=args.train_ratio,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
