"""Package an explicit source allowlist into a self-contained Kaggle notebook."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import re
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DATASET = "dmrashidpferrer/r2000-pestnet"


def cell(source, kind="code"):
    result = {"cell_type": kind, "metadata": {},
              "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(username="YOUR_KAGGLE_USERNAME"):
    # Never traverse the repository wholesale: venvs, data, credentials and results stay out.
    sources = [ROOT / name for name in ("train.py", "val.py", "predict.py", "verify_paddy_yolo.py", "pytest.ini", "requirements.txt")]
    sources += sorted((ROOT / "paddy_yolo").rglob("*.py"))
    sources += sorted((ROOT / "tests").glob("test_*.py"))
    sources += [ROOT / "configs" / name for name in ("paddy-yolo.yaml", "hyp_paper.yaml", "r2000.yaml")]
    # Enhanced modules are in the shared package; their tests and license need these files.
    enhanced_config = ROOT / "configs/paddy-yolo-ema-dysample.yaml"
    if enhanced_config.is_file():
        sources.append(enhanced_config)
        sources.append(ROOT / "third_party/dysample-LICENSE.txt")
        sources.append(ROOT / "kaggle/ema_dysample/runtime.py")
    sources += [OUTPUT / name for name in ("prepare_dataset.py", "run_baseline.py", "requirements.txt")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            content = source.read_bytes()
            if re.search(rb"KGAT_[A-Za-z0-9_\-]{16,}", content):
                raise ValueError(f"Refusing to bundle an API token in {source.relative_to(ROOT)}")
            archive.writestr(source.relative_to(ROOT).as_posix(), content)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "cells": [
            cell("""
                # Paddy-YOLO baseline on R2000-PestNet

                Attach **dmrashidpferrer/r2000-pestnet**, enable **GPU** and **Internet**, then run all cells.
                This notebook embeds the project source and pins Ultralytics to the locally verified version.
                It preserves the dataset's train/validation/test splits and generic class IDs.
                The default **one epoch** checks the pipeline; set `EPOCHS = 300` for a longer experiment.
                This is the Paddy-YOLO baseline (RGG + DIMIoU H). `MODEL_KIND = 'yolov8n'` selects the
                standard YOLOv8-N comparison with CIoU. A single supplied split is not five-fold validation.
                The attached dataset needs no API token inside this notebook.
            """, "markdown"),
            cell("""
                EPOCHS = 1
                BATCH = 16  # Reduce to 8 if GPU memory is insufficient.
                IMGSZ = 640
                WORKERS = 2
                MODEL_KIND = "paddy"
                SCRATCH = False  # False transfers matching COCO yolov8n.pt weights.
                RUN_CHECKS = True
            """),
            cell(f"""
                import base64, io, os, sys, zipfile
                from pathlib import Path
                PROJECT = Path('/kaggle/working/paddy_yolo_source')
                PROJECT.mkdir(parents=True, exist_ok=True)
                B64_SOURCE = '{encoded}'
                with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_SOURCE))) as archive:
                    archive.extractall(PROJECT)
                os.chdir(PROJECT)
                sys.path.insert(0, str(PROJECT))
                os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/baseline_settings/ultralytics'
                os.environ['MPLCONFIGDIR'] = '/kaggle/working/baseline_settings/matplotlib'
                os.environ['WANDB_MODE'] = 'disabled'
                print('Project extracted to', PROJECT)
            """),
            cell("""
                import subprocess
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                                str(PROJECT / 'kaggle/baseline/requirements.txt')], check=True)
                import torch, torchvision, ultralytics
                from ultralytics.utils import SETTINGS
                SETTINGS.update({key: False for key in
                                 ('raytune', 'wandb', 'comet', 'clearml', 'mlflow', 'neptune')})
                print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
                print('Ultralytics:', ultralytics.__version__, 'CUDA:', torch.cuda.is_available())
                assert torch.cuda.is_available(), 'Enable GPU acceleration in Kaggle notebook settings.'
                assert ultralytics.__version__ == '8.3.18'
                print('GPU:', torch.cuda.get_device_name(0))
                if RUN_CHECKS:
                    subprocess.run([sys.executable, 'verify_paddy_yolo.py'], cwd=PROJECT, check=True)
                    subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=PROJECT, check=True)
            """),
            cell("""
                import json
                sys.path.insert(0, str(PROJECT / 'kaggle/baseline'))
                from prepare_dataset import prepare_dataset
                DATA_YAML = prepare_dataset(Path('/kaggle/input'), Path('/kaggle/working/baseline_data'))
                print('Dataset ready:', DATA_YAML)
            """),
            cell("""
                OUTPUT_DIR = Path('/kaggle/working/baseline_output')
                command = [sys.executable, str(PROJECT / 'kaggle/baseline/run_baseline.py'),
                           '--data', str(DATA_YAML), '--output-dir', str(OUTPUT_DIR),
                           '--epochs', str(EPOCHS), '--batch', str(BATCH), '--imgsz', str(IMGSZ),
                           '--workers', str(WORKERS), '--device', '0', '--model-kind', MODEL_KIND]
                if SCRATCH:
                    command.append('--scratch')
                subprocess.run(command, cwd=PROJECT, check=True)
            """),
            cell("""
                print((OUTPUT_DIR / 'metrics.json').read_text())
                print('Download baseline_output/ for best.pt, last.pt, results.csv, plots and metrics.json.')
                print('Download baseline_data/dataset_audit.json for class counts and split checks.')
            """),
        ],
    }
    for index, entry in enumerate(notebook["cells"]):
        entry["id"] = f"baseline-{index:02d}"
    path = OUTPUT / "baseline.ipynb"
    path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": f"{username}/paddy-yolo-r2000-pestnet-baseline",
                "title": "Paddy-YOLO R2000-PestNet Baseline", "code_file": path.name,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}
    (OUTPUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {path.name}: {len(sources)} source files, {len(buffer.getvalue()):,} archive bytes.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="YOUR_KAGGLE_USERNAME", help="Your Kaggle username for optional API push")
    build_notebook(parser.parse_args().username)
