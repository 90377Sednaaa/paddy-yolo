"""Paddy-YOLO Inference and Prediction Script.

Runs object detection inference on single images, directories, or videos.
Annotates detections with bounding boxes and saves predictions.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional
import torch
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for running detection inference."""
    parser = argparse.ArgumentParser(
        description="Run Detection Inference using Paddy-YOLO"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/r2000/images",
        help="Path to input image, directory, or video (default: dataset/r2000/images)",
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
        help="Model architecture config YAML (used if weights not found, default: configs/paddy-yolo.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for bounding box filtering (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cpu', '0', 'cuda', etc. Default: auto-detect)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/predict",
        help="Project directory for saving predictions (default: runs/predict)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name for prediction output folder (default: exp)",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Disable saving annotated prediction images",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detection results in YOLO text format (*.txt)",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence scores in saved *.txt files",
    )
    parser.add_argument(
        "--save-crop",
        action="store_true",
        help="Save cropped bounding box images",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated images in window",
    )
    return parser.parse_args(args)


def predict(args: argparse.Namespace) -> List[Any]:
    """Run Paddy-YOLO inference."""
    register_paddy_modules()

    # Determine model source
    model_source: str
    if args.weights is not None and Path(args.weights).is_file():
        model_source = args.weights
    elif args.model is not None and Path(args.model).is_file():
        model_source = args.model
    elif args.weights is not None:
        model_source = args.weights
    else:
        model_source = "configs/paddy-yolo.yaml"

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Paddy-YOLO Inference")
    print("=" * 60)
    print(f"Model Source   : {model_source}")
    print(f"Input Source   : {args.source}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Confidence     : {args.conf}")
    print(f"IoU Threshold  : {args.iou}")
    print(f"Device         : {device}")
    print(f"Save Detections: {args.save}")
    print("=" * 60)

    model = YOLO(model_source)

    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
    )

    total_detections = sum(len(r.boxes) for r in results)
    save_dir = results[0].save_dir if results and hasattr(results[0], "save_dir") else None

    print("\n" + "=" * 60)
    print("Paddy-YOLO Prediction Summary")
    print("=" * 60)
    print(f"Total Images Processed : {len(results)}")
    print(f"Total Objects Detected : {total_detections}")
    if save_dir:
        print(f"Results saved to       : {save_dir}")
    print("=" * 60)
    print("Prediction completed successfully.")

    return results


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    predict(args)


if __name__ == "__main__":
    main()
