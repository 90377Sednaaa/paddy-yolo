"""Train YOLOv8-N from scratch and evaluate all five R2000 test folds."""

import argparse
import csv
import gc
import hashlib
import json
import statistics
import math
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from paddy_yolo.dataset.prepare_r2000 import prepare_kaggle_r2000, create_5fold_splits, R2000_CLASSES


def check_training_history(path, expected_epochs=300):
    """Require complete, finite epoch history before publishing test results."""
    with Path(path).open(encoding="utf-8") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(file)]
    if [int(row["epoch"]) for row in rows] != list(range(1, expected_epochs + 1)):
        raise ValueError(f"Incomplete or repeated epoch history: {path}")
    for row in rows:
        bad = {key: value for key, value in row.items() if not math.isfinite(float(value))}
        if bad:
            raise FloatingPointError(f"Non-finite epoch {row['epoch']} in {path}: {bad}")
    return {"epochs": len(rows), "all_values_finite": True}


def run(raw_data, output, experiment="baseline"):
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline replication requires a CUDA GPU")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_kaggle_r2000(raw_data, output / "dataset")
    folds = create_5fold_splits(data, seed=42)
    images = list((data / "images").iterdir())
    labels = list((data / "labels").glob("*.txt"))
    boxes = sum(len(p.read_text().strip().splitlines()) for p in labels)
    if (len(images), boxes) != (2024, 2281):
        raise ValueError(f"Expected 2024 images / 2281 boxes; got {len(images)} / {boxes}")
    is_impdiou = experiment == "rgg-impdiou"
    is_dimiou = experiment == "rgg-dimiou"
    is_paddy = is_impdiou or is_dimiou
    run_name = experiment.replace("-", "_") if is_paddy else "baseline_yolov8n"
    manifest = {
        "status": "running", "model": "YOLOv8-N", "box_loss": "CIoU",
        "initialization": "random weights; no pretrained checkpoint", "epochs": 300,
        "batch": 32, "imgsz": 640, "seed": 42, "optimizer": "AdamW",
        "training_precision": "AMP", "validation_precision": "FP32",
        "warmup_bias_lr": 0.0, "finite_epoch_checks": True,
        "dataset": "dmrashidpferrer/r2000-pestnet", "images": len(images), "boxes": boxes,
        "split": "fivefold: 3 train / 1 validation / 1 test; byte-identical images grouped",
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0), "completed_folds": [],
        "hyperparameters_sha256": hashlib.sha256(Path("configs/hyp_paper.yaml").read_bytes()).hexdigest(),
        "fold_membership_sha256": {
            str(i): {role: hashlib.sha256("\n".join(sorted(
                Path(line).name for line in (data / f"fold{i}_{role}.txt").read_text().splitlines()
            )).encode()).hexdigest() for role in ("train", "val", "test")} for i in range(5)
        },
    }
    if is_paddy:
        manifest.update(model="Paddy-YOLO RGG backbone", box_loss="IMPDIoU",
                        fixed_ratio=1.3, criterion="PaddyDetectionLoss / PaddyBboxLoss",
                        architecture="configs/paddy-yolo.yaml", expected_rgg_blocks=4,
                        paper_reference="Table 13: + RGG + IMPD (mAP50 69.7%, mAP50:95 50.8%)")
        if is_dimiou:
            manifest.update(box_loss="DIMIoU", fixed_ratio=None, strategy="H",
                            paper_reference="Tables 8/13: Paddy-YOLO + RGG + DIM (mAP50 70.2%, mAP50:95 51.5%)")
        from paddy_yolo.engine.trainer import PaddyTrainer
        from paddy_yolo.modules.rgg import RGG
        trainer = PaddyTrainer.__new__(PaddyTrainer)
        trainer.data = {"nc": len(R2000_CLASSES), "channels": 3}
        trainer.strategy, trainer.fixed_ratio = "H", (1.3 if is_impdiou else None)
        trainer.focal_gamma, trainer.focal_alpha = 0.0, 0.75
        model = trainer.get_model(cfg="configs/paddy-yolo.yaml", verbose=False)
        criterion = model.init_criterion()
        rgg_indices = [i for i, module in enumerate(model.model) if isinstance(module, RGG)]
        assert rgg_indices == [2, 4, 6, 8]
        assert all(model.model[i].f == -1 for i in rgg_indices)
        manifest["rgg_stage_indices"] = rgg_indices
        assert criterion.bbox_loss.fixed_ratio == trainer.fixed_ratio
        assert criterion.bbox_loss.strategy == "H"
        manifest["criterion_preflight_verified"] = True
        del model, criterion, trainer
    progress = output / "run_manifest.json"
    rows = []
    try:
        for fold, data_yaml in enumerate(folds):
            manifest["current_fold"] = fold
            progress.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Starting {run_name} fold {fold}: 300 epochs", flush=True)
            command = [sys.executable, "train.py", "--data", data_yaml,
                       "--hyp", "configs/hyp_paper.yaml", "--epochs", "300", "--batch", "32",
                       "--imgsz", "640", "--device", "0", "--seed", "42", "--workers", "2",
                       "--project", str(output / "train"), "--name", f"{run_name}_fold{fold}"]
            if is_impdiou:
                command += ["--model", "configs/paddy-yolo.yaml", "--impdiou", "--inner-ratio", "1.3"]
            elif is_dimiou:
                command += ["--model", "configs/paddy-yolo.yaml", "--strategy", "H"]
            else:
                command.append("--baseline")
            with (output / f"fold{fold}_training.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                if process.wait() != 0:
                    raise RuntimeError(f"Fold {fold} failed; see its training log")
            weights = output / "train" / f"{run_name}_fold{fold}" / "weights/best.pt"
            history = check_training_history(weights.parent.parent / "results.csv")
            model = YOLO(str(weights))
            metrics = model.val(data=data_yaml, split="test", imgsz=640, batch=32, device="0",
                                project=str(output / "test"), name=f"fold{fold}", plots=True)
            p, r = float(metrics.box.mp), float(metrics.box.mr)
            row = {"fold": fold, "precision": p, "recall": r, "f1": 2*p*r/(p+r+1e-16),
                   "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map)}
            per_class = {R2000_CLASSES[int(cls)]: dict(zip(
                ("precision", "recall", "mAP50", "mAP50_95"), map(float, metrics.box.class_result(i))))
                         for i, cls in enumerate(metrics.box.ap_class_index)}
            (output / f"fold{fold}_test_metrics.json").write_text(
                json.dumps({**row, "per_class": per_class, "history_check": history}, indent=2), encoding="utf-8")
            rows.append(row)
            with (output / "individual_folds.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            manifest["completed_folds"].append(fold)
            progress.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            del model, metrics
            gc.collect()
            torch.cuda.empty_cache()
        summary = {key: {"mean": statistics.mean(row[key] for row in rows),
                         "sample_std": statistics.stdev(row[key] for row in rows)}
                   for key in rows[0] if key != "fold"}
        (output / "fivefold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        print("Fivefold baseline complete:", json.dumps(summary), flush=True)
    except Exception:
        manifest["status"] = "error"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        progress.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--output", default="runs/baseline_crossval")
    parser.add_argument("--experiment", choices=["baseline", "rgg-impdiou", "rgg-dimiou"], default="baseline")
    args = parser.parse_args()
    run(args.raw_data, args.output, args.experiment)
