import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_fold0_config_d_warmup_bias_0_lr_001_kernel")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Zip local codebase
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if d not in [
                ".git", "__pycache__", ".pytest_cache", ".superpowers",
                "docs", "runs", "data", "weights", "kaggle_kernel", "kaggle_paddy_kernel",
                "kaggle_enhanced_kernel", "kaggle_ema_800_kernel",
                "kaggle_residual_ema_kernel", "kaggle_5fold_ema_kernel",
                "kaggle_fold0_ema_kernel", "kaggle_fold0_ema_warmup_bias_0_kernel",
                "kaggle_fold0_ema_warmup_bias_0_lr_001_kernel",
                "kaggle_fold0_dysample_kernel", "kaggle_fold0_dysample_warmup_bias_0_kernel",
                "kaggle_fold0_dysample_warmup_bias_0_lr_001_kernel", "kaggle_5fold_dysample_kernel",
                "kaggle_fold0_ema_dysample_kernel", "kaggle_5fold_ema_dysample_kernel",
                "kaggle_fold0_config_d_warmup_bias_0_kernel", "kaggle_fold0_config_d_warmup_bias_0_lr_001_kernel",
                "kaggle_5fold_scratch_kernel", "kaggle_5fold_baseline_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Fold 0 Paddy-YOLO Configuration D (EMA + DySample) with lr0 = 0.001 and warmup_bias_lr = 0.0
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Fold 0 Benchmark: Paddy-YOLO + EMA + DySample (Config D) with Calibrated AdamW (lr0 = 0.001, warmup_bias_lr = 0.0)\n",
                "\n",
                "This notebook executes training and evaluation of **Paddy-YOLO Configuration D** (4 RGG modules + 3 EMA Attention Blocks at P3, P4, P5 + 2 DySample Dynamic Upsamplers in Neck) on **Fold 0** of the **R2000** rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Experimental Hyperparameter Configuration:\n",
                "- **Architecture**: `configs/paddy-yolo-ema-dysample.yaml` (RGG backbone + EMA attention at P3, P4, P5 + DySample at neck layers 13 and 16)\n",
                "- **Bounding Box Loss**: DIMIoU with Strategy H dynamic ratio scheduling\n",
                "- **Weight Initialization**: COCO topological pre-training transfer (`yolov8n.pt`)\n",
                "- **Training Schedule**: 300 epochs, batch size 32, image size 640x640, AdamW optimizer\n",
                "- **Calibrated Learning Rate**: **`lr0 = 0.001`**, `lrf = 0.01` (10x lower learning rate suited for AdamW to prevent parameter divergence)\n",
                "- **Warmup Configuration**: `warmup_epochs = 3.0`, `warmup_momentum = 0.8`, **`warmup_bias_lr = 0.0`** (zero bias warmup to eliminate early gradient shock)\n",
                "- **Target Run Directory**: `paddy_yolo_config_d_fold0_warmup_bias_0_lr_001`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1: Extract project codebase\n",
                "import base64, io, zipfile, os\n",
                "\n",
                f"B64_ARCHIVE = '''{b64_code}'''\n",
                "\n",
                "zip_data = base64.b64decode(B64_ARCHIVE.encode('utf-8'))\n",
                "with zipfile.ZipFile(io.BytesIO(zip_data)) as z:\n",
                "    z.extractall('.')\n",
                "\n",
                "print('Codebase successfully unpacked into working directory.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2: Verify GPU Accelerator & PyTorch Environment\n",
                "import torch\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device:', torch.cuda.get_device_name(0))\n",
                "    print('Device Count:', torch.cuda.device_count())\n",
                "    print('VRAM Available:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB')\n",
                "else:\n",
                "    raise SystemError('GPU accelerator is not available! Please enable GPU in notebook settings.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3: Install Ultralytics and Cache yolov8n.pt\n",
                "!pip install ultralytics -q\n",
                "!python verify_paddy_yolo.py --smoke-test\n",
                "\n",
                "from ultralytics import YOLO\n",
                "# Pre-download base weights for instant transfer\n",
                "base_model = YOLO('yolov8n.pt')\n",
                "print('Base yolov8n.pt successfully cached.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Discover & Prepare Kaggle R2000 Dataset with 5-Fold Splits\n",
                "import os, subprocess\n",
                "from pathlib import Path\n",
                "from paddy_yolo.dataset.prepare_r2000 import prepare_kaggle_r2000, create_5fold_splits\n",
                "\n",
                "kaggle_input = Path('/kaggle/input')\n",
                "data_yamls = list(kaggle_input.rglob('data.yaml')) if kaggle_input.exists() else []\n",
                "\n",
                "if data_yamls:\n",
                "    raw_data_dir = data_yamls[0].parent\n",
                "    print(f'Discovered R2000 dataset via data.yaml at: {raw_data_dir}')\n",
                "else:\n",
                "    imgs = list(kaggle_input.rglob('*.jpg')) if kaggle_input.exists() else []\n",
                "    if imgs:\n",
                "        raw_data_dir = imgs[0].parent.parent\n",
                "        print(f'Discovered R2000 dataset via image search at: {raw_data_dir}')\n",
                "    else:\n",
                "        print('Dataset not mounted in /kaggle/input. Downloading via Kaggle API...')\n",
                "        download_dir = Path('/kaggle/working/data/raw_r2000')\n",
                "        download_dir.mkdir(parents=True, exist_ok=True)\n",
                "        subprocess.run([\n",
                "            'kaggle', 'datasets', 'download',\n",
                "            '-d', 'dmrashidferrer/r2000v1-2024-images',\n",
                "            '--unzip', '-p', str(download_dir)\n",
                "        ], check=True)\n",
                "        raw_data_dir = download_dir\n",
                "        print(f'Downloaded and unzipped dataset to: {raw_data_dir}')\n",
                "\n",
                "output_dir = Path('/kaggle/working/data/r2000')\n",
                "prepare_kaggle_r2000(input_dir=raw_data_dir, output_dir=output_dir)\n",
                "folds = create_5fold_splits(dataset_dir=output_dir)\n",
                "print('Generated 5-fold cross validation split configs:')\n",
                "for f in folds:\n",
                "    print(f'  -> {f}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Train Paddy-YOLO Config D (EMA + DySample) Fold 0 with lr0 = 0.001 and warmup_bias_lr = 0.0\n",
                "import time, os, subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "start_time = time.time()\n",
                "data_yaml = '/kaggle/working/data/r2000/data_fold0.yaml'\n",
                "run_name = 'paddy_yolo_config_d_fold0_warmup_bias_0_lr_001'\n",
                "\n",
                "print('=' * 85)\n",
                "print('STARTING TRAINING: Paddy-YOLO Config D Fold 0 (lr0 = 0.001, warmup_bias_lr = 0.0)')\n",
                "print('=' * 85)\n",
                "\n",
                "# Train 300 epochs with COCO topological transfer and RAM caching\n",
                "cmd_train = [\n",
                "    'python', '-u', 'train.py',\n",
                "    '--model', 'configs/paddy-yolo-ema-dysample.yaml',\n",
                "    '--weights', 'yolov8n.pt',\n",
                "    '--data', data_yaml,\n",
                "    '--hyp', 'configs/hyp_warmup_bias_0_lr_001.yaml',\n",
                "    '--epochs', '300',\n",
                "    '--batch', '32',\n",
                "    '--imgsz', '640',\n",
                "    '--cache', 'True',\n",
                "    '--device', '0',\n",
                "    '--project', '/kaggle/working/runs/train',\n",
                "    '--name', run_name\n",
                "]\n",
                "print(f'Executing Training Command: {\" \".join(cmd_train)}')\n",
                "subprocess.run(cmd_train, check=True)\n",
                "\n",
                "elapsed_train = (time.time() - start_time) / 3600.0\n",
                "print(f'Training completed in {elapsed_train:.2f} hours.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Robust Checkpoint Discovery, Evaluation & Comparative Synthesis\n",
                "import pandas as pd, numpy as np, subprocess\n",
                "from pathlib import Path\n",
                "from ultralytics import YOLO\n",
                "from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES\n",
                "\n",
                "run_name = 'paddy_yolo_config_d_fold0_warmup_bias_0_lr_001'\n",
                "data_yaml = '/kaggle/working/data/r2000/data_fold0.yaml'\n",
                "\n",
                "# Dynamically locate best.pt across potential nested output directories\n",
                "candidate_paths = [\n",
                "    Path(f'/kaggle/working/runs/train/{run_name}/weights/best.pt'),\n",
                "    Path(f'/kaggle/working/runs/detect/runs/train/{run_name}/weights/best.pt'),\n",
                "    Path(f'/kaggle/working/runs/detect/train/{run_name}/weights/best.pt'),\n",
                "    Path(f'/kaggle/working/runs/{run_name}/weights/best.pt'),\n",
                "]\n",
                "best_ckpt = None\n",
                "for cp in candidate_paths:\n",
                "    if cp.exists():\n",
                "        best_ckpt = str(cp)\n",
                "        break\n",
                "\n",
                "if best_ckpt is None:\n",
                "    found = list(Path('/kaggle/working').rglob('best.pt'))\n",
                "    if found:\n",
                "        best_ckpt = str(found[0])\n",
                "    else:\n",
                "        raise FileNotFoundError(f'Could not locate best.pt for {run_name}')\n",
                "\n",
                "print(f'Discovered best model checkpoint at: {best_ckpt}')\n",
                "\n",
                "# Run val.py on unseen holdout test split\n",
                "cmd_val = [\n",
                "    'python', '-u', 'val.py',\n",
                "    '--weights', best_ckpt,\n",
                "    '--data', data_yaml,\n",
                "    '--split', 'test',\n",
                "    '--imgsz', '640',\n",
                "    '--device', '0'\n",
                "]\n",
                "print(f'Executing Test Evaluation: {\" \".join(cmd_val)}')\n",
                "subprocess.run(cmd_val, check=True)\n",
                "\n",
                "# Detailed metrics extraction\n",
                "eval_model = YOLO(best_ckpt)\n",
                "metrics = eval_model.val(\n",
                "    data=data_yaml,\n",
                "    split='test',\n",
                "    imgsz=640,\n",
                "    batch=32,\n",
                "    device=0,\n",
                "    plots=True\n",
                ")\n",
                "\n",
                "p_all = float(metrics.box.mp) * 100.0\n",
                "r_all = float(metrics.box.mr) * 100.0\n",
                "map50_all = float(metrics.box.map50) * 100.0\n",
                "map50_95_all = float(metrics.box.map) * 100.0\n",
                "f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-6)\n",
                "\n",
                "# Reference values (calibrated AdamW, lr0=0.001, warmup_bias_0):\n",
                "paper_p, paper_r, paper_map50, paper_map50_95, paper_f1 = 73.10, 63.00, 70.20, 51.50, 67.70\n",
                "base_p, base_r, base_map50, base_map50_95, base_f1 = 77.59, 71.17, 74.90, 58.11, 74.24\n",
                "dys_p, dys_r, dys_map50, dys_map50_95, dys_f1 = 74.30, 72.04, 75.47, 59.05, 73.15\n",
                "\n",
                "summary_data = [\n",
                "    {\n",
                "        'Metric': 'Precision (%)',\n",
                "        'Paper (Table 8)': f'{paper_p:.2f}%',\n",
                "        'Baseline Fold 0 (lr=0.001)': f'{base_p:.2f}%',\n",
                "        'DySample Fold 0 (lr=0.001)': f'{dys_p:.2f}%',\n",
                "        'Config D (EMA+DySample)': f'{p_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{p_all - base_p:+.2f}%',\n",
                "        'Delta vs Paper': f'{p_all - paper_p:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'Recall (%)',\n",
                "        'Paper (Table 8)': f'{paper_r:.2f}%',\n",
                "        'Baseline Fold 0 (lr=0.001)': f'{base_r:.2f}%',\n",
                "        'DySample Fold 0 (lr=0.001)': f'{dys_r:.2f}%',\n",
                "        'Config D (EMA+DySample)': f'{r_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{r_all - base_r:+.2f}%',\n",
                "        'Delta vs Paper': f'{r_all - paper_r:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50 (%)',\n",
                "        'Paper (Table 8)': f'{paper_map50:.2f}%',\n",
                "        'Baseline Fold 0 (lr=0.001)': f'{base_map50:.2f}%',\n",
                "        'DySample Fold 0 (lr=0.001)': f'{dys_map50:.2f}%',\n",
                "        'Config D (EMA+DySample)': f'{map50_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{map50_all - base_map50:+.2f}%',\n",
                "        'Delta vs Paper': f'{map50_all - paper_map50:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50-95 (%)',\n",
                "        'Paper (Table 8)': f'{paper_map50_95:.2f}%',\n",
                "        'Baseline Fold 0 (lr=0.001)': f'{base_map50_95:.2f}%',\n",
                "        'DySample Fold 0 (lr=0.001)': f'{dys_map50_95:.2f}%',\n",
                "        'Config D (EMA+DySample)': f'{map50_95_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{map50_95_all - base_map50_95:+.2f}%',\n",
                "        'Delta vs Paper': f'{map50_95_all - paper_map50_95:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'F1 Score (%)',\n",
                "        'Paper (Table 8)': f'{paper_f1:.2f}%',\n",
                "        'Baseline Fold 0 (lr=0.001)': f'{base_f1:.2f}%',\n",
                "        'DySample Fold 0 (lr=0.001)': f'{dys_f1:.2f}%',\n",
                "        'Config D (EMA+DySample)': f'{f1_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{f1_all - base_f1:+.2f}%',\n",
                "        'Delta vs Paper': f'{f1_all - paper_f1:+.2f}%'\n",
                "    }\n",
                "]\n",
                "\n",
                "df_summary = pd.DataFrame(summary_data)\n",
                "print('\\n' + '=' * 105)\n",
                "print('PADDY-YOLO CONFIG D (EMA + DYSAMPLE) FOLD 0 BENCHMARK (lr0=0.001, warmup_bias_lr=0.0)')\n",
                "print('=' * 105)\n",
                "print(df_summary.to_string(index=False))\n",
                "df_summary.to_csv('paddy_yolo_config_d_fold0_warmup_bias_0_lr_001_vs_paper.csv', index=False)\n",
                "\n",
                "# Per-species table\n",
                "p_cls = metrics.box.p * 100.0\n",
                "r_cls = metrics.box.r * 100.0\n",
                "ap50_cls = metrics.box.ap50 * 100.0\n",
                "ap_cls = metrics.box.ap * 100.0\n",
                "\n",
                "species_rows = []\n",
                "for idx, name in enumerate(R2000_CLASSES):\n",
                "    if idx < len(p_cls):\n",
                "        species_rows.append({\n",
                "            'Class ID': idx,\n",
                "            'Species': name,\n",
                "            'P (%)': round(float(p_cls[idx]), 2),\n",
                "            'R (%)': round(float(r_cls[idx]), 2),\n",
                "            'mAP@50 (%)': round(float(ap50_cls[idx]), 2),\n",
                "            'mAP@50-95 (%)': round(float(ap_cls[idx]), 2)\n",
                "        })\n",
                "\n",
                "df_species = pd.DataFrame(species_rows)\n",
                "print('\\n' + '=' * 90)\n",
                "print('PER-SPECIES PERFORMANCE (CONFIG D FOLD 0 Calibrated AdamW)')\n",
                "print('=' * 90)\n",
                "print(df_species.to_string(index=False))\n",
                "df_species.to_csv('paddy_yolo_config_d_fold0_warmup_bias_0_lr_001_species_results.csv', index=False)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Safe & Resilient Artifact Archiving\n",
                "import shutil\n",
                "from pathlib import Path\n",
                "\n",
                "run_name = 'paddy_yolo_config_d_fold0_warmup_bias_0_lr_001'\n",
                "\n",
                "# Search for run output directory\n",
                "run_dirs = [\n",
                "    Path(f'/kaggle/working/runs/train/{run_name}'),\n",
                "    Path(f'/kaggle/working/runs/detect/runs/train/{run_name}'),\n",
                "    Path(f'/kaggle/working/runs/detect/train/{run_name}'),\n",
                "    Path(f'/kaggle/working/runs/{run_name}'),\n",
                "    Path('/kaggle/working/runs/train'),\n",
                "    Path('/kaggle/working/runs')\n",
                "]\n",
                "archive_src = None\n",
                "for d in run_dirs:\n",
                "    if d.exists() and (d / 'weights').exists():\n",
                "        archive_src = d\n",
                "        break\n",
                "\n",
                "if archive_src is None:\n",
                "    for d in run_dirs:\n",
                "        if d.exists():\n",
                "            archive_src = d\n",
                "            break\n",
                "\n",
                "if archive_src:\n",
                "    dest_zip = '/kaggle/working/paddy_yolo_config_d_fold0_warmup_bias_0_lr_001_results'\n",
                "    print(f'Compressing run artifacts from: {archive_src} -> {dest_zip}.zip')\n",
                "    zip_path = shutil.make_archive(dest_zip, 'zip', str(archive_src))\n",
                "    print(f'✅ All weights and results compressed successfully into: {zip_path}')\n",
                "else:\n",
                "    print('⚠️ Warning: No specific run directory found for archiving; CSVs remain safely in /kaggle/working.')\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = out_dir / "paddy_yolo_config_d_fold0_warmup_bias_0_lr_001.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook saved to: {nb_path} ({nb_path.stat().st_size} bytes)")

# 3. Kernel metadata for Kaggle
meta = {
    "id": "sednaaa90377/paddy-yolo-config-d-fold0-warmup-bias-0-lr-001",
    "title": "paddy-yolo-config-d-fold0-warmup-bias-0-lr-001",
    "code_file": "paddy_yolo_config_d_fold0_warmup_bias_0_lr_001.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_gpu": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "dmrashidferrer/r2000v1-2024-images"
    ],
    "competition_sources": [],
    "kernel_sources": []
}

meta_path = out_dir / "kernel-metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Kernel metadata saved to: {meta_path}")
print("Paddy-YOLO Config D Fold 0 (warmup_bias_0 + lr0=0.001) Kaggle Kernel Builder completed successfully!")
