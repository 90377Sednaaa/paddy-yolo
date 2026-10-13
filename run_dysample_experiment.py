"""Prepare one R2000 fold and train either Paddy-YOLO architecture on Kaggle.

This script is deliberately not called by the preflight notebook unless its
RUN_FULL_TRAINING switch is changed to True by the user.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from paddy_yolo.dataset.prepare_r2000 import create_5fold_splits, prepare_kaggle_r2000


CONFIGS = {
    "baseline": "configs/paddy-yolo.yaml",
    "proposed": "configs/Paddy-YOLO-DySample.yaml",
    "proposed_plus": "configs/Paddy-YOLO-DySample-Plus.yaml",
    "ema_heads": "configs/Paddy-YOLO-EMA-Heads.yaml",
    "p3_only": "configs/Paddy-YOLO-DySample-P3.yaml",
}


def validate_resume_checkpoint(checkpoint: str | Path, variant: str, fold: int) -> None:
    """Reject checkpoints whose restored Ultralytics arguments would change the experiment."""
    if variant not in CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = saved.get("train_args") or {}
    architecture = getattr(saved.get("ema"), "yaml", None) or {}
    head = architecture.get("head", [])
    config = Path(__file__).resolve().parent / CONFIGS[variant]
    expected_head = yaml.safe_load(config.read_text(encoding="utf-8"))["head"]
    if len(head) < 4 or any(head[index][2:] != expected_head[index][2:] for index in (0, 3)):
        raise ValueError(f"Resume checkpoint architecture does not match {variant}")
    if (Path(str(args.get("data", ""))).name != f"data_fold{fold}.yaml"
            or args.get("name") != f"fold{fold}"):
        raise ValueError(f"Resume checkpoint fold does not match fold {fold}")
    if Path(str(args.get("project", ""))).name != variant:
        raise ValueError(f"Resume checkpoint project does not match {variant}")
    if saved.get("optimizer") is None or saved.get("scaler") is None or saved.get("epoch", -1) < 0:
        raise ValueError("Resume checkpoint is missing optimizer, scaler, or epoch state")


def training_command(variant: str, fold: int, data: str | Path,
                     output_root: str | Path, gpu_count: int,
                     resume_from: str | None = None) -> list[str]:
    """Use one synchronized DDP run with a global batch of 32 on two GPUs."""
    if variant not in CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")
    if fold not in range(5):
        raise ValueError(f"Fold must be 0-4: {fold}")
    if gpu_count < 1:
        raise RuntimeError("At least one CUDA GPU is required")
    command = [
        sys.executable, "train.py", "--model", CONFIGS[variant],
        "--data", str(data), "--hyp", "configs/hyp_paper.yaml",
        "--epochs", "300", "--batch", "32", "--imgsz", "640",
        "--device", "0,1" if gpu_count >= 2 else "0",
        "--save-period", "50", "--workers", "2", "--seed", "42",
        "--project", str(Path(output_root) / variant), "--name", f"fold{fold}",
    ]
    if resume_from:
        command += ["--resume-from", resume_from]
    else:
        command += ["--weights", "yolov8n.pt"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/dysample_training"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    if args.resume_from:
        validate_resume_checkpoint(args.resume_from, args.variant, args.fold)
    dataset = prepare_kaggle_r2000(args.raw_data, Path("/kaggle/working/data/r2000"))
    folds = create_5fold_splits(dataset, seed=42)
    command = training_command(args.variant, args.fold, folds[args.fold],
                               args.output_root, torch.cuda.device_count(),
                               str(args.resume_from) if args.resume_from else None)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
