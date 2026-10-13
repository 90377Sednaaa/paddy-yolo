"""Train Paddy-YOLO variants independently on five R2000 folds."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES, create_5fold_splits, prepare_kaggle_r2000
from paddy_yolo.engine.baseline import FP32DetectionValidator
from paddy_yolo.modules.dysample import DySample
from paddy_yolo.modules.ema import EMA
from paddy_yolo.modules.rgg import RGG
from run_baseline_crossval import check_training_history
from run_dysample_experiment import CONFIGS, training_command, validate_resume_checkpoint


METRICS = ("precision", "recall", "mAP50", "mAP50_95")
VARIANT = "proposed"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_scores(values):
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise FloatingPointError(f"Invalid detection metrics: {values}")


def export_metrics(metrics, fold, images_seen):
    """Export exact validator values; unsupported classes get empty metrics."""
    scores = [float(v) for v in (metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map)]
    finite_scores(scores)
    overall = {"fold": fold, "class_id": "", "class_name": "overall", "images": int(images_seen),
               "instances": int(sum(metrics.nt_per_class)), **dict(zip(METRICS, scores))}
    indices = {int(cls): index for index, cls in enumerate(metrics.box.ap_class_index)}
    classes = []
    for cls, name in enumerate(R2000_CLASSES):
        values = [None] * 4
        if cls in indices:
            values = [float(v) for v in metrics.box.class_result(indices[cls])]
            finite_scores(values)
        classes.append({"fold": fold, "class_id": cls, "class_name": name,
                        "images": int(metrics.nt_per_image[cls]), "instances": int(metrics.nt_per_class[cls]),
                        **dict(zip(METRICS, values))})
    return {"overall": overall, "per_class": classes}


def evaluate_checkpoint(checkpoint, data_yaml, output, fold, device="0", variant=VARIANT):
    """Evaluate the validation-selected checkpoint on the held-out test split."""
    model = YOLO(str(checkpoint))
    if variant == "ema_heads":
        layers = model.model.model
        if ([i for i, layer in enumerate(layers) if isinstance(layer, EMA)] != [22, 23, 24]
                or any(isinstance(layer, DySample) for layer in layers)
                or [layers[i].f for i in (22, 23, 24)] != [15, 18, 21]
                or layers[-1].f != [22, 23, 24]
                or [i for i, layer in enumerate(layers) if isinstance(layer, RGG)] != [2, 4, 6, 8]):
            raise ValueError("Test checkpoint is not the EMA detection-branch architecture")
    elif variant == "p3_only":
        layers = model.model.model
        layer = layers[13]
        if ([i for i, module in enumerate(layers) if isinstance(module, DySample)] != [13]
                or not isinstance(layers[10], torch.nn.Upsample) or layers[10].mode != "nearest"
                or hasattr(layer, "scope") or (layer.scale, layer.style, layer.groups) != (2, "lp", 4)
                or layer.offset.in_channels != 128
                or any(isinstance(module, EMA) for module in layers)
                or [i for i, module in enumerate(layers) if isinstance(module, RGG)] != [2, 4, 6, 8]
                or layers[-1].f != [15, 18, 21] or layers[-1].nc != 16):
            raise ValueError("Test checkpoint is not the P3-only DySample architecture")
    else:
        for index in (10, 13):
            layer = model.model.model[index]
            if (not isinstance(layer, DySample) or hasattr(layer, "scope")
                    or (layer.scale, layer.style, layer.groups) != (2, "lp", 4)):
                raise ValueError("Test checkpoint is not the plain DySample architecture")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    validator = None
    try:
        validator = FP32DetectionValidator(save_dir=Path(output) / "test" / f"fold{fold}", args={
            "data": str(data_yaml), "split": "test", "imgsz": 640, "batch": 32,
            "device": device, "half": False, "plots": True, "workers": 2,
            "project": str(Path(output) / "test"), "name": f"fold{fold}",
        })
        validator(model=model.model)
        return export_metrics(validator.metrics, fold, validator.seen)
    finally:
        # select_device('0') changes the environment. Subsequent training
        # subprocesses must inherit the original visibility of both GPUs.
        if visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
        del model, validator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate_results(results, output):
    """Compute an unweighted five-fold mean and sample standard deviation."""
    if len(results) != 5 or sorted(r["overall"]["fold"] for r in results) != list(range(5)):
        raise ValueError("A five-fold summary requires five unique completed folds, 0 through 4")
    output = Path(output)
    summary = {}
    for metric in METRICS:
        values = [r["overall"][metric] for r in results]
        finite_scores(values)
        summary[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}
    classes = []
    for cls, name in enumerate(R2000_CLASSES):
        records = [next(row for row in r["per_class"] if row["class_id"] == cls) for r in results]
        supported = [row for row in records if row["mAP50"] is not None]
        row = {"class_id": cls, "class_name": name, "evaluated_folds": len(supported)}
        for metric in METRICS:
            values = [r[metric] for r in supported]
            finite_scores(values)
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        classes.append(row)
    write_json(output / "fivefold_summary.json", summary)
    write_csv(output / "fivefold_summary.csv", [{"metric": metric, **values} for metric, values in summary.items()])
    write_csv(output / "per_class_summary.csv", classes)
    return summary


def run_training(command, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)


def validate_restored_resume(checkpoint, output, fold, variant=VARIANT):
    """Bind a checkpoint to restored experiment settings and its saved history."""
    checkpoint = Path(checkpoint).resolve()
    fold_dir = Path(output) / variant / f"fold{fold}"
    if checkpoint.parent != (fold_dir / "weights").resolve():
        raise ValueError("Restore the checkpoint inside its original five-fold training directory")
    history = fold_dir / "results.csv"
    best = fold_dir / "weights/best.pt"
    if not history.is_file() or not best.is_file():
        raise ValueError("Restore the interrupted fold's results.csv and best.pt before resuming")
    validate_resume_checkpoint(checkpoint, variant, fold)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    root = Path(__file__).resolve().parent
    expected = yaml.safe_load((root / "configs/hyp_paper.yaml").read_text())
    expected.update(seed=42, workers=2, project=str(Path(output) / variant), name=f"fold{fold}")
    settings = saved["train_args"]
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Resume checkpoint training settings do not match the five-fold experiment")
    config = yaml.safe_load((root / CONFIGS[variant]).read_text())
    architecture = saved["ema"].yaml
    if any(architecture.get(key) != value for key, value in config.items()):
        raise ValueError(f"Resume checkpoint full architecture does not match {variant}")
    snapshot = saved.get("train_results")
    length = saved["epoch"] + 1
    if (not isinstance(snapshot, dict) or snapshot.get("epoch") != list(range(1, length + 1))
            or any(len(values) != length for values in snapshot.values())):
        raise ValueError("Resume checkpoint has incomplete saved epoch history")
    with history.open(encoding="utf-8") as handle:
        rows = [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if len(rows) < length or not rows or set(rows[0]) != set(snapshot):
        raise ValueError("Restored CSV does not match checkpoint history")
    check_training_history(history, expected_epochs=len(rows))
    if any(not math.isclose(rows[i][key], float(values[i]), rel_tol=0, abs_tol=1e-10)
           for key, values in snapshot.items() for i in range(length)):
        raise ValueError("Restored CSV belongs to different checkpoint history")
    previous_best = torch.load(best, map_location="cpu", weights_only=False)
    best_fitness = previous_best.get("train_metrics", {}).get("fitness")
    if (best_fitness is None or saved.get("best_fitness") is None
            or not math.isclose(float(best_fitness), float(saved["best_fitness"]), abs_tol=1e-5)):
        raise ValueError("Restore the best.pt associated with this checkpoint, not a later best checkpoint")
    restored = [{key: values[i] for key, values in snapshot.items()} for i in range(length)]
    for row in restored:
        row["epoch"] = int(row["epoch"])
    return restored


def run(raw_data, output, start_fold=0, resume_from=None, variant=VARIANT):
    if variant not in ("proposed", "ema_heads"):
        raise ValueError(f"Unsupported five-fold variant: {variant}")
    model_label = "EMA detection branches" if variant == "ema_heads" else "plain DySample"
    if not 0 <= start_fold < 5:
        raise ValueError("start_fold must be 0 through 4")
    if not torch.cuda.is_available():
        raise RuntimeError("Full five-fold training requires a CUDA GPU")
    gpu_count = torch.cuda.device_count()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = prepare_kaggle_r2000(raw_data, output / "dataset")
    folds = create_5fold_splits(dataset, seed=42)
    root = Path(__file__).resolve().parent
    fingerprint = {
        "architecture_sha256": hashlib.sha256((root / CONFIGS[variant]).read_bytes()).hexdigest(),
        "hyperparameters_sha256": hashlib.sha256((root / "configs/hyp_paper.yaml").read_bytes()).hexdigest(),
        "fold_membership_sha256": {
            str(i): {role: hashlib.sha256("\n".join(sorted(Path(line).name for line in
                         (dataset / f"fold{i}_{role}.txt").read_text().splitlines())).encode()).hexdigest()
                     for role in ("train", "val", "test")} for i in range(5)
        },
    }
    previous = {}
    progress = output / "run_manifest.json"
    if progress.exists():
        previous = json.loads(progress.read_text())
        if any(previous.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Existing five-fold outputs have different architecture, hyperparameters or splits")
    results = []
    for fold in range(start_fold):
        if fold not in previous.get("completed_folds", []):
            raise ValueError(f"Missing completed fold {fold}; restore previous run outputs before continuing")
        result = json.loads((output / f"fold{fold}_test_metrics.json").read_text())
        if result["overall"]["fold"] != fold:
            raise ValueError(f"Saved fold result does not match fold {fold}")
        results.append(result)
    restored_history = None
    if resume_from:
        if not previous:
            raise ValueError("Restore the original five-fold output tree and manifest before resuming")
        restored_history = validate_restored_resume(resume_from, output, start_fold, variant)
    ready_for_test = set()
    for fold in range(start_fold, 5):
        fold_dir = output / variant / f"fold{fold}"
        if fold_dir.exists() and not (resume_from and fold == start_fold):
            if previous and fold == start_fold and fold not in previous.get("completed_folds", []):
                try:
                    check_training_history(fold_dir / "results.csv", expected_epochs=300)
                    if not (fold_dir / "weights/best.pt").is_file():
                        raise FileNotFoundError("Best checkpoint missing")
                    ready_for_test.add(fold)
                    continue
                except (ValueError, FloatingPointError, FileNotFoundError):
                    pass
            raise FileExistsError(f"Fold {fold} training directory already exists; use a fresh output or resume explicitly")
    manifest = {
        "status": "running", "variant": variant, "architecture": CONFIGS[variant],
        "initialization": "yolov8n.pt independently for every fresh fold", "epochs": 300, "batch": 32,
        "imgsz": 640, "seed": 42, "save_period": 50, "test_split": "held-out fifth; 3 train / 1 val / 1 test",
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "completed_folds": list(range(start_fold)), **fingerprint,
    }
    if variant == "ema_heads":
        manifest.update(attention="project residual Efficient Multi-Scale Attention", attention_groups=8,
                        attention_indices=[22, 23, 24], attention_sources=[15, 18, 21],
                        upsampling="nearest", box_loss="DIMIoU", strategy="H", fixed_ratio=None)
    else:
        manifest["dyscope"] = False
    if resume_from:
        manifest["resume_from"] = str(resume_from)
    try:
        for fold in range(start_fold, 5):
            manifest["current_fold"] = fold
            write_json(progress, manifest)
            if fold in ready_for_test:
                print(f"Retrying test evaluation for completed training fold {fold}", flush=True)
            else:
                if restored_history is not None and fold == start_fold:
                    write_csv(output / variant / f"fold{fold}" / "results.csv", restored_history)
                command = training_command(variant, fold, folds[fold], output, gpu_count,
                                           str(resume_from) if resume_from and fold == start_fold else None)
                print(f"Starting {model_label} fold {fold}: {' '.join(command)}", flush=True)
                run_training(command, output / f"fold{fold}_training.log")
            checkpoint = output / variant / f"fold{fold}" / "weights/best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}")
            history = check_training_history(checkpoint.parent.parent / "results.csv", expected_epochs=300)
            if variant == VARIANT:
                result = evaluate_checkpoint(checkpoint, folds[fold], output, fold)
            else:
                result = evaluate_checkpoint(checkpoint, folds[fold], output, fold, variant=variant)
            result["history_check"] = history
            write_json(output / f"fold{fold}_test_metrics.json", result)
            results.append(result)
            write_csv(output / "individual_folds.csv", [r["overall"] for r in results])
            write_csv(output / "per_class_and_overall.csv", [row for r in results
                       for row in [r["overall"], *r["per_class"]]])
            manifest["completed_folds"].append(fold)
            write_json(progress, manifest)
        summary = aggregate_results(results, output)
        manifest["status"] = "complete"
        print(f"{model_label} five-fold test results:", json.dumps(summary), flush=True)
        return summary
    except Exception:
        manifest["status"] = "error"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        write_json(progress, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/dysample_fivefold"))
    parser.add_argument("--start-fold", type=int, choices=range(5), default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--variant", choices=["proposed", "ema_heads"], default=VARIANT)
    args = parser.parse_args()
    run(args.raw_data, args.output, args.start_fold, args.resume_from, args.variant)
