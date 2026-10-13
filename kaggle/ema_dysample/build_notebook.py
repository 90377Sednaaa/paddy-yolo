"""Build the private single-run EMA + DySample Kaggle notebook and metadata."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kaggle/baseline"))
from build_notebook import cell


def build(username="dmrashidpferrer"):
    config = json.loads((OUTPUT / "run_config.json").read_text())
    config["account"] = username
    sources = [ROOT / name for name in ("train.py", "val.py", "predict.py", "verify_paddy_yolo.py",
                                        "requirements.txt", "pytest.ini")]
    sources += sorted((ROOT / "paddy_yolo").rglob("*.py"))
    sources += sorted((ROOT / "tests").glob("test_*.py"))
    sources += [ROOT / "configs" / name for name in ("paddy-yolo.yaml", "paddy-yolo-ema-dysample.yaml",
                                                    "hyp_paper.yaml", "hyp_ema_dysample.yaml", "r2000.yaml")]
    sources += [ROOT / "RGG_EMA_DySample_changes.txt"]
    sources += [ROOT / "kaggle/baseline" / name for name in ("prepare_dataset.py", "run_baseline.py", "requirements.txt")]
    sources += [OUTPUT / "run_training.py", OUTPUT / "requirements.txt",
                OUTPUT / "runtime.py", OUTPUT / "run_config.json",
                ROOT / "third_party/dysample-LICENSE.txt"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            content = source.read_bytes()
            if re.search(rb"KGAT_[A-Za-z0-9_\-]{16,}", content):
                raise ValueError(f"Refusing to bundle an API token: {source.relative_to(ROOT)}")
            archive.writestr(source.relative_to(ROOT).as_posix(), content)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}}, "cells": [
        cell("""
            # Paddy YOLO EMA DySample RGG Fold0

            **300 epochs**, nano scale, 640 pixels, total batch 32 across two GPUs (16 each),
            `yolov8n.pt` initialization.
            AdamW uses `lr0=0.001` and `warmup_bias_lr=0.0` for this corrected RGG rerun.
            RGG convenience aliases no longer register duplicate state tensors during weight averaging.
            Three EMA blocks follow the P3/P4/P5 RGG blocks; two DySample layers replace FPN upsampling.
            Bounding-box training uses DIMIoU H, with BCE classification and DFL.
            Dataset: `dmrashidpferrer/r2000-pestnet`. This single experiment preserves its supplied
            train/validation/test split; `fold0` is a run label, not a regenerated cross-validation fold.
            Compatible pretrained tensors are transferred by corresponding layer, accounting for index shifts.
            New EMA/DySample parameters and incompatible COCO classifier tensors are initialized separately.
            GPU and Internet are enabled in the accompanying private notebook metadata.
        """, "markdown"),
        cell(f"""
            import base64, io, os, sys, zipfile
            from pathlib import Path
            PROJECT = Path('/kaggle/working/paddy_yolo_ema_dysample_source')
            PROJECT.mkdir(parents=True, exist_ok=True)
            B64_SOURCE = '{encoded}'
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_SOURCE))) as archive:
                archive.extractall(PROJECT)
            os.chdir(PROJECT)
            sys.path.insert(0, str(PROJECT))
            os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/ema_dysample_settings/ultralytics'
            os.environ['MPLCONFIGDIR'] = '/kaggle/working/ema_dysample_settings/matplotlib'
            os.environ['WANDB_MODE'] = 'disabled'
            os.environ['PYTHONUNBUFFERED'] = '1'
            print('Extracted project:', PROJECT)
        """),
        cell("""
            import json, subprocess
            RUN = json.loads((PROJECT / 'kaggle/ema_dysample/run_config.json').read_text())
            print(json.dumps(RUN, indent=2))
            sys.path.insert(0, str(PROJECT / 'kaggle/ema_dysample'))
            from runtime import configure_ultralytics, run_logged
            run_logged([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                        str(PROJECT / 'kaggle/ema_dysample/requirements.txt')],
                       Path('/kaggle/working/environment_setup.log'), PROJECT)
            import torch, torchvision, ultralytics
            assert torch.cuda.is_available(), 'This run requires a Kaggle GPU.'
            assert ultralytics.__version__ == '8.3.18'
            configure_ultralytics()
            print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
            print('GPU:', torch.cuda.get_device_name(0))
            run_logged([sys.executable, '-m', 'pytest', '-q', '--tb=short'],
                       Path('/kaggle/working/preflight_tests.log'), PROJECT)
        """),
        cell("""
            sys.path.insert(0, str(PROJECT / 'kaggle/baseline'))
            from prepare_dataset import prepare_dataset
            DATA = prepare_dataset(Path('/kaggle/input'), Path('/kaggle/working/ema_dysample_data'))
            # Download official COCO YOLOv8-N weights; no Kaggle credentials are needed here.
            from ultralytics import YOLO
            pretrained = YOLO(RUN['initial_weights'])
            print('Initial weights ready:', RUN['initial_weights'])
            del pretrained
        """),
        cell("""
            OUTPUT = Path('/kaggle/working/ema_dysample_output')
            command = [sys.executable, str(PROJECT / 'kaggle/ema_dysample/run_training.py'),
                       '--data', str(DATA), '--output-dir', str(OUTPUT),
                       '--epochs', str(RUN['epochs']), '--batch', str(RUN['batch']),
                       '--imgsz', str(RUN['imgsz']), '--workers', str(RUN['workers']),
                       '--device', RUN['device'], '--weights', RUN['initial_weights'],
                       '--hyp', str(PROJECT / RUN['hyp']), '--name', RUN['run_name'],
                       '--patience', str(RUN['patience']), '--save-period', str(RUN['save_period'])]
            run_logged(command, OUTPUT / 'training.log', PROJECT)
        """),
        cell("""
            print((OUTPUT / 'metrics.json').read_text())
            print((OUTPUT / 'weight_transfer.json').read_text())
            print('Download ema_dysample_output for checkpoints, results.csv and validation/test plots.')
            print('Dataset audit:', DATA.parent / 'dataset_audit.json')
        """),
    ]}
    for i, entry in enumerate(notebook["cells"]):
        entry["id"] = f"ema-dysample-{i:02d}"
    notebook_path = OUTPUT / "ema_dysample.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": f"{username}/{config['kernel_slug']}", "title": config['title'],
                "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [config["dataset"]], "competition_sources": [], "kernel_sources": []}
    (OUTPUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {notebook_path.name}: {len(sources)} files, {len(buffer.getvalue()):,} archive bytes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="dmrashidpferrer")
    build(parser.parse_args().username)
