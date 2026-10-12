"""Paddy-YOLO Validation and Evaluation Script.

Evaluates trained Paddy-YOLO checkpoints or model configurations on validation or test sets.
Reports mAP@50, mAP@50:95, Precision, Recall, and F1-score across classes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Paddy-YOLO on Validation or Test Sets"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights (.pt checkpoint)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/paddy-yolo.yaml",
        help="Model architecture config YAML (used if weights not specified, default: configs/paddy-yolo.yaml)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/r2000/data_fold0.yaml",
        help="Path to prepared dataset YAML (default: dataset/r2000/data_fold0.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for validation (default: 32)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate ('val', 'test', or 'train', default: 'val')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run validation on ('cpu', '0', 'cuda', etc. Default: auto-detect)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for evaluation (default: 0.001)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold for evaluation (default: 0.6)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/val",
        help="Project directory for saving evaluation outputs (default: runs/val)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="paddy_yolo_val",
        help="Experiment name for evaluation run (default: paddy_yolo_val)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save results in COCO-format JSON",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save evaluation plots (confusion matrix, PR curves)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Test-Time Augmentation (TTA) multi-scale inference",
    )
    return parser.parse_args(args)


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    """Run validation and display key metrics."""
    register_paddy_modules()

    # Determine model source: weights file or architecture yaml
    model_source: str
    if args.weights is not None:
        weights_path = Path(args.weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights file not found: {args.weights}")
        model_source = str(weights_path)
    elif args.model is not None:
        model_path = Path(args.model)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model config file not found: {args.model}")
        model_source = str(model_path)
    else:
        model_source = "configs/paddy-yolo.yaml"

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Paddy-YOLO Model Evaluation")
    print("=" * 60)
    print(f"Model Source   : {model_source}")
    print(f"Dataset Config : {args.data}")
    print(f"Split          : {args.split}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Batch Size     : {args.batch}")
    print(f"Device         : {device}")
    print(f"Augment (TTA)  : {args.augment}")
    print("=" * 60)

    model = YOLO(model_source)
    # Fuse custom RepConv branches before Ultralytics fuses their child Convs.
    from paddy_yolo.modules.repconv import RepConv
    model.model.eval()
    for module in list(model.model.modules()):
        if isinstance(module, RepConv):
            module.switch_to_deploy()

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        plots=args.plots,
        augment=args.augment,
    )

    # Extract metrics safely
    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0
    mp = float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0
    mr = float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0
    f1 = float(2 * mp * mr / (mp + mr + 1e-16))

    print("\n" + "=" * 60)
    print("Paddy-YOLO Evaluation Summary")
    print("=" * 60)
    print(f"mAP@50     : {map50:.4f}")
    print(f"mAP@50:95  : {map50_95:.4f}")
    print(f"Precision  : {mp:.4f}")
    print(f"Recall     : {mr:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print("=" * 60)
    print("Validation completed successfully.")

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": mp,
        "recall": mr,
        "f1": f1,
        "metrics": metrics,
    }


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    validate(args)


if __name__ == "__main__":
    main()
