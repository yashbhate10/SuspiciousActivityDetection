"""
Enterprise YOLOv11 Crime Classifier Training v3.

Fixes for v2's 34% accuracy:
  1. Use yolo11m-cls (medium) for better feature extraction capacity
  2. Train at 224px (standard ImageNet size — well-optimized in pretrained weights)
  3. 50 epochs with patience=15 (give the model more room to converge)
  4. Stronger augmentation: MixUp + Mosaic disabled (classification mode),
     but color jitter, erasing, and geometric transforms enabled
  5. Learning rate warmup via Ultralytics defaults
  6. Label smoothing to prevent overconfident predictions
  7. Cosine LR schedule for smoother convergence
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "anomaly_cls_v3"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    base_model: str = "yolo11m-cls.pt",   # medium >> small for scene classification
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 224,                     # standard pretrained resolution
    device: str = "auto",
    patience: int = 15,
):
    print("=" * 65)
    print("  ENTERPRISE CRIME CLASSIFIER TRAINING v3")
    print(f"  Dataset : {data_dir}")
    print(f"  Model   : {base_model}")
    print(f"  Epochs  : {epochs}  |  Batch: {batch}  |  ImgSz: {imgsz}")
    print("=" * 65)

    if not data_dir.exists() or not (data_dir / "train").exists():
        print(f"\n  ERROR: Dataset not found at {data_dir}")
        print("  Run first: python ai-engine/prepare_dataset.py")
        sys.exit(1)

    # Count classes and frames
    classes = sorted([d.name for d in (data_dir / "train").iterdir() if d.is_dir()])
    print(f"\n  Classes ({len(classes)}): {classes}")
    for cls in classes:
        train_count = len(list((data_dir / "train" / cls).glob("*.jpg")))
        val_count = len(list((data_dir / "val" / cls).glob("*.jpg")))
        print(f"    {cls:<10} train={train_count:>5}  val={val_count:>5}")

    if "Normal" not in classes:
        print("\n  CRITICAL: No 'Normal' class found!")
        print("  The model will ALWAYS predict a crime. Run prepare_dataset.py first.")
        sys.exit(1)

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "0" and torch.cuda.is_available() else "CPU"
    print(f"\n  Device: {gpu_name}")

    model = YOLO(base_model)

    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name="crime_clf_v3",
        exist_ok=True,
        plots=True,
        verbose=True,
        # ── Optimiser ──
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,          # final LR = lr0 * lrf (cosine decay)
        weight_decay=0.01,
        warmup_epochs=3,
        # ── Regularisation ──
        label_smoothing=0.1,    # prevents overconfident wrong predictions
        dropout=0.2,            # classification head dropout
        # ── Augmentation (moderate — training data is small) ──
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.15,
        scale=0.2,
        shear=2.0,
        perspective=0.0001,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        erasing=0.3,        # random erasing (cutout)
    )

    # Copy best weights
    best = PROJECT_ROOT / "runs" / "classify" / "crime_clf_v3" / "weights" / "best.pt"
    target = MODELS_DIR / "anomaly_yolo11_best.pt"
    if best.exists():
        shutil.copy(best, target)
        print(f"\n  Best weights saved to: {target}")
    else:
        print(f"\n  WARNING: Could not find best weights at {best}")

    # Validation metrics
    print("\n  Evaluating on validation set...")
    val_model = YOLO(str(target if target.exists() else best))
    metrics = val_model.val(data=str(data_dir), imgsz=imgsz, device=device)

    top1 = metrics.top1
    top5 = metrics.top5

    print("\n" + "=" * 65)
    print("  FINAL VALIDATION METRICS")
    print("=" * 65)
    print(f"  Top-1 Accuracy : {top1:.4f} ({top1 * 100:.2f}%)")
    print(f"  Top-5 Accuracy : {top5:.4f} ({top5 * 100:.2f}%)")
    print(f"  Weights saved  : {target}")
    print("=" * 65)

    if top1 >= 0.75:
        print("\n  ✅ Model accuracy is GOOD. Ready for deployment.")
    elif top1 >= 0.60:
        print("\n  ⚠️ Acceptable accuracy. Consider more data or epochs.")
    else:
        print("\n  ❌ Accuracy below 60%. Suggestions:")
        print("    1. More epochs: --epochs 80")
        print("    2. More frames: --crime-frames 50 in prepare_dataset.py")
        print("    3. Check dataset quality: are Normal frames truly normal?")
        print("    4. Try larger model: yolo11l-cls.pt (if GPU VRAM allows)")


def main():
    parser = argparse.ArgumentParser(description="Train crime classifier v3")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model", default="yolo11m-cls.pt",
                        help="yolo11s-cls, yolo11m-cls, yolo11l-cls")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16, help="Reduce to 8 if CUDA OOM")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    train(
        data_dir=Path(args.data),
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
