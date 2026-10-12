"""Paddy-YOLO Training Script.

Executes training using PaddyTrainer with DIMIoU loss (Strategy H) and paper hyperparameters.
Supports 5-fold cross validation fold selection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import torch

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.engine import PaddyTrainer


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for training Paddy-YOLO."""
    parser = argparse.ArgumentParser(
        description="Train Paddy-YOLO with DIMIoU Loss and Paper Hyperparameters"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Train standard baseline YOLOv8-N model with CIoU loss instead of Paddy-YOLO with DIMIoU",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/r2000.yaml",
        help="Path to dataset YAML config (default: configs/r2000.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/paddy-yolo.yaml",
        help="Path to model architecture YAML config (default: configs/paddy-yolo.yaml)",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="configs/hyp_paper.yaml",
        help="Path to hyperparameters YAML config (default: configs/hyp_paper.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides hyp file)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (overrides hyp file)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size in pixels (overrides hyp file)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="Fold index (0-4) for 5-fold cross validation split",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="H",
        help="Dynamic ratio scaling strategy for DIMIoU loss (default: 'H')",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focusing parameter gamma for Focal Loss (default: 0.0 [disabled, standard BCE], set >0 to enable)",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help="Positive/negative class balance parameter alpha for Focal Loss (default: 0.75)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="MixUp data augmentation probability (overrides hyp file)",
    )
    parser.add_argument(
        "--copy-paste",
        type=float,
        default=None,
        help="Copy-Paste data augmentation probability (overrides hyp file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cpu', '0', 'cuda', etc. Default: auto-detect)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional initial model weights path (.pt)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory for training artifacts (default: runs/train)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="paddy_yolo",
        help="Experiment name (default: paddy_yolo)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader worker threads (default: 4 on Linux, 0 on Windows)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from most recent checkpoint",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="plots",
        help="Disable generation of evaluation plots and confusion matrices",
    )
    parser.add_argument(
        "--no-val",
        action="store_false",
        dest="val",
        help="Disable validation during training",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save",
        help="Disable saving model checkpoints",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache dataset images in RAM ('ram', 'True') or disk ('disk') for faster training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable detailed logging output",
    )
    return parser.parse_args(args)


def resolve_data_config(data: str, fold: Optional[int] = None) -> str:
    """Resolve data YAML path, accounting for 5-fold cross validation if requested."""
    if fold is not None:
        candidate_paths = [
            Path(f"dataset/r2000/data_fold{fold}.yaml"),
            Path(f"data_fold{fold}.yaml"),
            Path(data).parent / f"data_fold{fold}.yaml",
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate.resolve())
        # If specific candidate file wasn't found directly, return the fold yaml name
        return f"data_fold{fold}.yaml"
    return data


def train(args: argparse.Namespace) -> PaddyTrainer:
    """Run Paddy-YOLO training."""
    # Ensure custom modules are registered
    register_paddy_modules()

    # Load baseline paper hyperparameters if specified
    hyp_params: Dict[str, Any] = {}
    if args.hyp and Path(args.hyp).is_file():
        with open(args.hyp, "r", encoding="utf-8") as f:
            hyp_params = yaml.safe_load(f) or {}

    # Resolve data path considering fold selection
    data_path = resolve_data_config(args.data, args.fold)

    # Resolve device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve default workers
    if args.workers is None:
        workers = 0 if os.name == "nt" else 4
    else:
        workers = args.workers

    # Default model and name based on baseline flag
    default_model = "yolov8n.yaml" if args.baseline else "configs/paddy-yolo.yaml"
    chosen_model = args.model if args.model != "configs/paddy-yolo.yaml" or not args.baseline else default_model

    if not args.baseline and args.weights:
        weights_path = Path(args.weights)
        if (
            weights_path.name in ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
            or "yolov8" in weights_path.name
        ) and "paddy" not in weights_path.name:
            print(f"Initializing Paddy-YOLO architecture from {chosen_model} and transferring weights from {args.weights}...")
            from ultralytics import YOLO
            init_yolo = YOLO(chosen_model)
            init_yolo.load(args.weights)

            # Transfer weights for shifted neck layers in attention-augmented models
            model_stem = Path(chosen_model).stem
            shift_maps = {
                "paddy-yolo-ema": {16: 17, 18: 19, 19: 21, 21: 23},
                "paddy-yolo-p2-ema": {16: 20, 18: 22, 19: 24, 21: 26},
            }
            if model_stem in shift_maps:
                layer_map = shift_maps[model_stem]
                src_ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
                src_sd = src_ckpt["model"].state_dict() if isinstance(src_ckpt, dict) and "model" in src_ckpt else src_ckpt.state_dict()
                tgt_sd = init_yolo.model.state_dict()
                shifted_count = 0
                for src_key, src_val in src_sd.items():
                    if not src_key.startswith("model."):
                        continue
                    parts = src_key.split(".")
                    src_idx = int(parts[1])
                    if src_idx in layer_map:
                        tgt_key = f"model.{layer_map[src_idx]}." + ".".join(parts[2:])
                        if tgt_key in tgt_sd and tgt_sd[tgt_key].shape == src_val.shape:
                            tgt_sd[tgt_key].copy_(src_val.float())
                            shifted_count += 1
                init_yolo.model.load_state_dict(tgt_sd)
                print(f"Topological weight transfer: mapped {shifted_count} additional neck tensors for {model_stem}.")

            weights_dir = Path("weights")
            weights_dir.mkdir(parents=True, exist_ok=True)
            init_path = weights_dir / f"{model_stem}_init.pt"
            torch.save({"model": init_yolo.model}, init_path)
            print(f"Initialized checkpoint created at: {init_path}")
            model_path = str(init_path)
        else:
            model_path = args.weights
    elif args.weights:
        model_path = args.weights
    else:
        model_path = chosen_model

    default_name = "baseline_yolov8n" if args.baseline else "paddy_yolo"
    run_name = args.name if args.name != "paddy_yolo" or not args.baseline else default_name
    experiment_name = f"{run_name}_fold{args.fold}" if args.fold is not None else run_name

    # Construct trainer overrides
    overrides: Dict[str, Any] = {
        **hyp_params,
        "model": model_path,
        "data": data_path,
        "strategy": args.strategy,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "device": device,
        "project": args.project,
        "name": experiment_name,
        "workers": workers,
        "seed": args.seed,
        "resume": args.resume,
        "plots": args.plots,
        "val": args.val,
        "save": args.save,
        "verbose": args.verbose,
    }

    # Explicit CLI arguments override hyp file values
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.mixup is not None:
        overrides["mixup"] = args.mixup
    if args.copy_paste is not None:
        overrides["copy_paste"] = args.copy_paste
    if args.cache is not None:
        overrides["cache"] = True if args.cache.lower() in ("true", "ram") else args.cache

    title = "Baseline YOLOv8-N" if args.baseline else "Paddy-YOLO"
    print("=" * 60)
    print(f"{title} Training Configuration")
    print("=" * 60)
    print(f"Model Architecture : {overrides.get('model')}")
    print(f"Dataset Config     : {overrides.get('data')}")
    print(f"Epochs             : {overrides.get('epochs', 300)}")
    print(f"Batch Size         : {overrides.get('batch', 32)}")
    print(f"Image Size         : {overrides.get('imgsz', 640)}")
    if not args.baseline:
        print(f"DIMIoU Strategy    : {overrides.get('strategy', 'H')}")
        if args.focal_gamma > 0.0:
            print(f"Classification Loss: Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
        else:
            print("Classification Loss: Standard Binary Cross-Entropy (BCE)")
    else:
        print("Bounding Box Loss  : CIoU (Standard YOLOv8 Baseline)")
    if overrides.get("mixup"):
        print(f"MixUp Augmentation : {overrides.get('mixup')}")
    if overrides.get("copy_paste"):
        print(f"Copy-Paste Aug     : {overrides.get('copy_paste')}")
    print(f"Device             : {overrides.get('device')}")
    print(f"Project / Name     : {overrides.get('project')} / {overrides.get('name')}")
    print("=" * 60)

    if args.baseline:
        from ultralytics.models.yolo.detect import DetectionTrainer
        # Remove custom strategy key for standard DetectionTrainer
        overrides.pop("strategy", None)
        trainer = DetectionTrainer(overrides=overrides)
    else:
        trainer = PaddyTrainer(overrides=overrides)

    trainer.train()

    print("\nTraining completed successfully.")
    if hasattr(trainer, "best") and trainer.best and Path(trainer.best).exists():
        print(f"Best checkpoint saved at: {trainer.best}")
    if hasattr(trainer, "last") and trainer.last and Path(trainer.last).exists():
        print(f"Last checkpoint saved at: {trainer.last}")

    return trainer


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
