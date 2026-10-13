"""Build a private full R2000 training notebook for Paddy-YOLO-DySample."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from build_dysample_kaggle_kernel import _cell, source_archive


def build(owner: str = "dmrashidferrer", fold: int = 0,
          output: str | Path | None = None,
          variant: str = "proposed") -> Path:
    if fold not in range(5):
        raise ValueError("fold must be 0-4")
    if variant not in ("proposed", "proposed_plus", "p3_only"):
        raise ValueError(f"Unknown full-run variant: {variant}")
    plus = variant == "proposed_plus"
    p3_only = variant == "p3_only"
    model_name = "Paddy-YOLO-DySample-P3" if p3_only else ("Paddy-YOLO-DySample+" if plus else "Paddy-YOLO-DySample")
    output = Path(output or ("kaggle_dysample_p3_kernel" if p3_only else
                            ("kaggle_dysample_plus_kernel" if plus else "kaggle_dysample_full_kernel")))
    output.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(source_archive()).decode("ascii")
    setup = f'''import base64, io, os, subprocess, sys, zipfile
from pathlib import Path
os.chdir('/kaggle/working')
os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/.ultralytics'
os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, ['/kaggle/working', os.environ.get('PYTHONPATH')]))
os.environ['PYTHONUNBUFFERED'] = '1'
B64_ARCHIVE = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_ARCHIVE))) as archive:
    archive.extractall('/kaggle/working')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.3.253', '--quiet'], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Kaggle GPU accelerator is required')
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
'''
    training = f'''# Full 300-epoch R2000 fold {fold} training; no synthetic preflight in this notebook.
VARIANT = {variant!r}
FOLD = {fold}
RESUME_FROM = None  # Set to a mounted checkpoint path in a later Kaggle session.
configs = [p for p in Path('/kaggle/input').rglob('data.yaml')
           if 'r2000' in str(p).lower()]
if not configs:
    download_dir = Path('/kaggle/working/data/raw_r2000')
    download_dir.mkdir(parents=True, exist_ok=True)
    print('R2000 was not mounted; downloading the published Kaggle dataset', flush=True)
    subprocess.run(['kaggle', 'datasets', 'download',
                    '-d', 'dmrashidpferrer/r2000-pestnet',
                    '--unzip', '-p', str(download_dir)], check=True)
    configs = list(download_dir.rglob('data.yaml'))
if len(configs) != 1:
    raise RuntimeError(f'Expected one R2000 data.yaml; found {{configs}}')
raw_data = configs[0].parent
print('R2000 source:', raw_data, flush=True)
command = [sys.executable, 'run_dysample_experiment.py',
           '--variant', VARIANT, '--fold', str(FOLD), '--raw-data', str(raw_data)]
if RESUME_FROM:
    command += ['--resume-from', RESUME_FROM]
print('Starting full R2000 training:', VARIANT, 'fold', FOLD, flush=True)
subprocess.run(command, check=True)
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# Full {model_name} training on R2000, fold {fold}\n",
                "COCO-pretrained yolov8n.pt initialization; 300 epochs, global batch 32, 640px, "
                "AdamW and DIMIoU Strategy H. Two T4 GPUs use one DDP model when available.\n",
            ]},
            _cell(setup), _cell(training),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    if p3_only:
        notebook["cells"][-1] = _cell(training.replace("run_dysample_experiment.py", "run_dysample_p3_experiment.py"))
        notebook["cells"][0]["source"].append(
            "Only P4-to-P3 upsampling uses DySample (layer 13); P5-to-P4 keeps nearest-neighbor. "
            "One fold only; best.pt receives FP32 held-out test evaluation and exact per-class exports.\n")
    filename = ("paddy_yolo_dysample_p3_r2000.ipynb" if p3_only else
                ("paddy_yolo_dysample_plus_r2000.ipynb" if plus else "paddy_yolo_dysample_r2000.ipynb"))
    (output / filename).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    slug = f"paddy-yolo-dysample{'-p3-only' if p3_only else ('-plus' if plus else '')}-r2000-fold{fold}"
    metadata = {
        "id": f"{owner}/{slug}", "title": slug.replace("-", " "),
        "code_file": filename, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": ["dmrashidpferrer/r2000-pestnet"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="dmrashidferrer")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output folder (defaults to a separate folder for each variant)")
    parser.add_argument("--variant", choices=["proposed", "proposed_plus", "p3_only"], default="proposed")
    args = parser.parse_args()
    print(build(args.owner, args.fold, args.output, args.variant))
