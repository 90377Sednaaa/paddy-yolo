"""Run the Paddy-YOLO baseline and evaluate its best checkpoint on held-out data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/baseline_output"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", default="yolov8n.pt", help="COCO weights; use --scratch to disable transfer")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--model-kind", choices=("paddy", "yolov8n"), default="paddy")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/paddy-yolo.yaml")
    parser.add_argument("--hyp", type=Path, default=ROOT / "configs/hyp_paper.yaml")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--save-period", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.path.insert(0, str(ROOT))
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS

    # The requested artifacts are local CSV/plots/checkpoints, without tracker accounts.
    SETTINGS.update({key: False for key in
                     ("raytune", "wandb", "comet", "clearml", "mlflow", "neptune")})

    if ultralytics.__version__ != "8.3.18":
        raise RuntimeError("Install kaggle/baseline/requirements.txt before running this baseline.")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU acceleration, or explicitly pass --device cpu for a local test.")
    world_size = len(args.device.split(",")) if args.device != "cpu" else 1
    if args.device != "cpu" and torch.cuda.device_count() < world_size:
        raise RuntimeError(f"This run requires {world_size} visible GPUs.")
    if args.batch < world_size or args.batch % world_size:
        raise ValueError("Total batch size must be divisible by the number of training GPUs.")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    name = args.name or f"{args.model_kind}_baseline"
    run_dir = output / "train" / name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists at {run_dir}; choose a new --output-dir to preserve previous results.")
    command = [sys.executable, str(ROOT / "train.py"), "--data", str(args.data.resolve()),
               "--hyp", str(args.hyp.resolve()), "--epochs", str(args.epochs),
               "--batch", str(args.batch), "--imgsz", str(args.imgsz), "--device", args.device,
               "--workers", str(args.workers), "--project", str(output / "train"),
               "--name", name, "--seed", "42"]
    if args.model_kind == "yolov8n":
        command.append("--baseline")
    else:
        command.extend(["--model", str(args.model.resolve()), "--strategy", "H"])
    if not args.scratch:
        command.extend(["--weights", args.weights])
    if args.patience is not None:
        command.extend(["--patience", str(args.patience)])
    if args.save_period is not None:
        command.extend(["--save-period", str(args.save_period)])
    if args.resume:
        command.append("--resume")
    print("Training:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")

    # Training selects best.pt using validation; held-out test data is evaluated here.
    from val import parse_args, validate
    results = {}
    for split in ("val", "test"):
        metrics = validate(parse_args([
            "--weights", str(best), "--data", str(args.data.resolve()), "--split", split,
            "--batch", str(args.batch), "--imgsz", str(args.imgsz), "--device", args.device,
            "--project", str(output / "evaluate"), "--name", split, "--plots",
        ]))
        results[split] = {key: value for key, value in metrics.items() if key != "metrics"}
    report = {"model": args.model_kind, "epochs": args.epochs, "seed": 42,
              "architecture": str(args.model.resolve()), "run_name": name,
              "imgsz": args.imgsz, "batch": args.batch, "data": str(args.data.resolve()),
              "world_size": world_size, "batch_per_gpu": args.batch // world_size,
              "initial_weights": None if args.scratch else args.weights,
              "ultralytics": ultralytics.__version__, "torch": torch.__version__,
              "device": args.device, "best_checkpoint": str(best), "metrics": results}
    report_path = output / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    transfer_file = ROOT / "weights" / f"{args.model.stem}_transfer.json"
    if transfer_file.is_file() and not args.resume and not args.scratch:
        (output / "weight_transfer.json").write_bytes(transfer_file.read_bytes())
    print(f"Baseline complete. Metrics: {report_path}")


if __name__ == "__main__":
    main()
