"""Train one fresh P4-to-P3-only DySample fold and report held-out test metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch

from paddy_yolo.dataset.prepare_r2000 import create_5fold_splits, prepare_kaggle_r2000
from run_baseline_crossval import check_training_history
from run_dysample_crossval import (
    evaluate_checkpoint, run_training, validate_restored_resume, write_csv, write_json,
)
from run_dysample_experiment import CONFIGS, training_command

VARIANT = "p3_only"


def run(raw_data, output, fold=0, resume_from=None):
    if fold not in range(5):
        raise ValueError("fold must be 0 through 4")
    if not torch.cuda.is_available():
        raise RuntimeError("Full R2000 training requires a CUDA GPU")
    gpu_count = torch.cuda.device_count()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = prepare_kaggle_r2000(raw_data, output / "dataset")
    folds = create_5fold_splits(dataset, seed=42)
    root = Path(__file__).resolve().parent
    fingerprint = {
        "variant": VARIANT, "current_fold": fold,
        "architecture_sha256": hashlib.sha256((root / CONFIGS[VARIANT]).read_bytes()).hexdigest(),
        "hyperparameters_sha256": hashlib.sha256((root / "configs/hyp_paper.yaml").read_bytes()).hexdigest(),
        "fold_membership_sha256": {
            str(fold): {role: hashlib.sha256("\n".join(sorted(Path(line).name for line in
                       (dataset / f"fold{fold}_{role}.txt").read_text().splitlines())).encode()).hexdigest()
                       for role in ("train", "val", "test")},
        },
    }
    progress = output / "run_manifest.json"
    previous = json.loads(progress.read_text()) if progress.exists() else {}
    if previous and any(previous.get(key) != value for key, value in fingerprint.items()):
        raise ValueError("Restored outputs have different model, fold, hyperparameters or splits")
    folder = output / VARIANT / f"fold{fold}"
    restored_history = None
    training_complete = False
    if resume_from:
        if not previous:
            raise ValueError("Restore the original output tree and manifest before resuming")
        restored_history = validate_restored_resume(resume_from, output, fold, VARIANT)
    elif folder.exists():
        if not previous:
            raise FileExistsError("Training output already exists without its manifest")
        check_training_history(folder / "results.csv", expected_epochs=300)
        if not (folder / "weights/best.pt").is_file():
            raise FileNotFoundError("Restore the completed fold's best.pt before evaluation")
        training_complete = True
    manifest = {
        **fingerprint, "status": "running", "architecture": CONFIGS[VARIANT],
        "initialization": "fresh compatible yolov8n.pt weights; seeded before model construction",
        "epochs": 300, "batch": 32, "imgsz": 640, "seed": 42, "optimizer": "AdamW",
        "save_period": 50, "dysample_indices": [13], "nearest_upsample_indices": [10],
        "dyscope": False, "style": "lp", "groups": 4, "scale": 2,
        "box_loss": "DIMIoU", "strategy": "H", "fixed_ratio": None,
        "test_split": "held-out fifth; 3 train / 1 val / 1 test",
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "scheduled_folds": [fold], "completed_folds": [],
    }
    if resume_from:
        manifest["resume_from"] = str(resume_from)
    write_json(progress, manifest)
    try:
        if not training_complete:
            if restored_history is not None:
                write_csv(folder / "results.csv", restored_history)
            command = training_command(VARIANT, fold, folds[fold], output, gpu_count,
                                       str(resume_from) if resume_from else None)
            print(f"Starting P3-only DySample fold {fold}: {' '.join(command)}", flush=True)
            run_training(command, output / f"fold{fold}_training.log")
        else:
            print(f"Training already complete; evaluating fold {fold} test split", flush=True)
        checkpoint = folder / "weights/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}")
        history = check_training_history(folder / "results.csv", expected_epochs=300)
        result = evaluate_checkpoint(checkpoint, folds[fold], output, fold, variant=VARIANT)
        result["history_check"] = history
        write_json(output / f"fold{fold}_test_metrics.json", result)
        write_csv(output / "individual_folds.csv", [result["overall"]])
        write_csv(output / "per_class_and_overall.csv", [result["overall"], *result["per_class"]])
        manifest.update(status="complete", completed_folds=[fold])
        print("P3-only DySample held-out test results:", json.dumps(result["overall"]), flush=True)
        return result
    except Exception:
        manifest.update(status="error", error=traceback.format_exc())
        raise
    finally:
        write_json(progress, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[VARIANT], default=VARIANT)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--raw-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/dysample_p3_training"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    run(args.raw_data, args.output_root, args.fold, args.resume_from)
