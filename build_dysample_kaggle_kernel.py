"""Package a private DySample preflight notebook for the dmrashidferrer account."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


DEFAULT_SLUG = "paddy-yolo-dysample-preflight"


def source_archive() -> bytes:
    """Bundle only reviewed project sources, with no checkpoints or credentials."""
    root = Path(__file__).resolve().parent
    files = [
        root / "train.py", root / "run_dysample_experiment.py",
        root / "run_dysample_crossval.py", root / "run_baseline_crossval.py",
        root / "verify_dysample.py", root / "configs/hyp_paper.yaml",
        root / "configs/paddy-yolo.yaml", root / "configs/Paddy-YOLO-DySample.yaml",
        root / "configs/Paddy-YOLO-DySample-Plus.yaml",
        root / "configs/Paddy-YOLO-EMA-Heads.yaml",
        root / "configs/Paddy-YOLO-DySample-P3.yaml", root / "run_dysample_p3_experiment.py",
        root / "paddy_yolo/modules/DySample-LICENSE",
        root / "tests/test_dysample.py", root / "tests/test_dysample_training_cli.py",
    ]
    files.extend((root / "paddy_yolo").rglob("*.py"))
    files = sorted(set(files))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package sources: {missing}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return buffer.getvalue()


def _cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def build(owner: str, output: Path | str = "kaggle_dysample_kernel",
          slug: str = DEFAULT_SLUG) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    encoded = base64.b64encode(source_archive()).decode("ascii")
    extraction = f'''import base64, io, os, zipfile
os.chdir('/kaggle/working')
os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/.ultralytics'
os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, ['/kaggle/working', os.environ.get('PYTHONPATH')]))
B64_ARCHIVE = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_ARCHIVE))) as archive:
    archive.extractall('/kaggle/working')
print('Paddy-YOLO sources unpacked')
'''
    validation = '''import subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.3.253',
                'pytest', '--quiet'], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Enable a Kaggle GPU accelerator to run this preflight')
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_dysample.py',
                'tests/test_dysample_training_cli.py', '-q'], check=True)
subprocess.run([sys.executable, 'verify_dysample.py', '--device', 'cuda:0',
                '--output', '/kaggle/working/dysample_verification',
                '--mini-train', '--mini-ddp'], check=True)
print('Preflight report: /kaggle/working/dysample_verification/comparison.json')
'''
    training = '''# Full training is opt-in. The submitted notebook runs only the validation above.
RUN_FULL_TRAINING = False
VARIANT = 'proposed'  # 'baseline' uses the original Paddy-YOLO architecture.
FOLD = 0
RESUME_FROM = None  # e.g. '/kaggle/input/prior-run/last.pt' for the SAME variant/fold.
if RUN_FULL_TRAINING:
    import subprocess, sys
    from pathlib import Path
    raw_configs = [p for p in Path('/kaggle/input').rglob('data.yaml')
                   if 'r2000' in str(p).lower()]
    if len(raw_configs) != 1:
        raise RuntimeError(f'Expected one mounted R2000 data.yaml, found {raw_configs}')
    command = [sys.executable, 'run_dysample_experiment.py', '--variant', VARIANT,
               '--fold', str(FOLD), '--raw-data', str(raw_configs[0].parent)]
    if RESUME_FROM:
        command += ['--resume-from', RESUME_FROM]
    subprocess.run(command, check=True)
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Paddy-YOLO-DySample: GPU preflight and optional R2000 training\n",
                "Baseline and proposed models both initialize compatible layers from yolov8n.pt. "
                "Only the two neck upsampling layers differ. Full training is disabled by default.\n",
            ]},
            _cell(extraction), _cell(validation), _cell(training),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }
    filename = "paddy_yolo_dysample_preflight.ipynb"
    (output / filename).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}", "title": "Paddy-YOLO DySample preflight",
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
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--output", default="kaggle_dysample_kernel")
    args = parser.parse_args()
    print(build(args.owner, args.output, args.slug))
