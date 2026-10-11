import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_enhanced_kernel")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Zip local codebase (excluding unwanted directories and artifacts)
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs 
            if d not in [
                ".git", "__pycache__", ".pytest_cache", ".superpowers", 
                "docs", "runs", "data", "kaggle_kernel", "kaggle_paddy_kernel", 
                "kaggle_enhanced_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created. Size: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Paddy-YOLO-EMA with Class-Balanced Focal Loss + TTA
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Proposed Enhanced Paddy-YOLO: EMA Attention + Class-Balanced Focal Loss + TTA\n",
                "\n",
                "This notebook trains and evaluates the enhanced **Paddy-YOLO-EMA** model with **Class-Balanced Focal Loss** and **Test-Time Augmentation (TTA)** on the **R2000** rice pest dataset (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Architectural & Loss Innovations:\n",
                "1. **Efficient Multi-Scale Attention (EMA, ICASSP 2023)**: 3 attention blocks before detection heads to filter leaf camouflage.\n",
                "2. **Class-Balanced Focal Loss (Lin et al., ICCV 2017)**: Adaptive focusing factor ($\\gamma=1.5, \\alpha=0.75$) to prevent rare pest classes (*asiatic rice borer*, *white backed plant hopper*) from being overwhelmed by frequent classes.\n",
                "3. **Dynamic Inner-MPDIoU (DIMIoU Strategy H)**: Angle and distance constrained regression loss.\n",
                "4. **Preserved 3-Head Balance**: 8,400 anchors preventing negative dilution.\n",
                "5. **Dual Evaluation**: Standard Single-Scale and Multi-Scale Test-Time Augmentation (TTA)."
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
                "# Cell 3: Install Ultralytics and Run Full Sanity Check\n",
                "!pip install ultralytics -q\n",
                "!python verify_paddy_yolo.py --smoke-test\n"
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
                "# Cell 5: Train Paddy-YOLO-EMA with Class-Balanced Focal Loss (300 Epochs)\n",
                "!python train.py \\\n",
                "    --model configs/paddy-yolo-ema.yaml \\\n",
                "    --weights yolov8n.pt \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --hyp configs/hyp_paper.yaml \\\n",
                "    --epochs 300 \\\n",
                "    --batch 32 \\\n",
                "    --imgsz 640 \\\n",
                "    --strategy H \\\n",
                "    --focal-gamma 1.5 \\\n",
                "    --focal-alpha 0.75 \\\n",
                "    --device 0 \\\n",
                "    --project /kaggle/working/runs/train \\\n",
                "    --name paddy_yolo_focal_ema_fold0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Comprehensive Dual Evaluation (Standard Single-Scale + Multi-Scale TTA)\n",
                "import os\n",
                "best_weights = '/kaggle/working/runs/train/paddy_yolo_focal_ema_fold0/weights/best.pt'\n",
                "\n",
                "print('=' * 70)\n",
                "print('1. STANDARD EVALUATION ON VALIDATION SPLIT (Fold 0 Val)')\n",
                "print('=' * 70)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split val --device 0\n",
                "\n",
                "print('\\n' + '=' * 70)\n",
                "print('2. STANDARD EVALUATION ON TEST SPLIT (Fold 0 Test)')\n",
                "print('=' * 70)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split test --device 0\n",
                "\n",
                "print('\\n' + '=' * 70)\n",
                "print('3. MULTI-SCALE TTA EVALUATION ON TEST SPLIT (Fold 0 Test + Augment)')\n",
                "print('=' * 70)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split test --augment --device 0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Thesis Comparative Benchmark Analysis\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "csv_path = Path('/kaggle/working/runs/train/paddy_yolo_focal_ema_fold0/results.csv')\n",
                "if csv_path.exists():\n",
                "    df = pd.read_csv(csv_path)\n",
                "    df.columns = df.columns.str.strip()\n",
                "    print('Training completed. Final 5 epochs metrics:')\n",
                "    cols = [c for c in ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'] if c in df.columns]\n",
                "    print(df[cols].tail(5).to_string())\n",
                "    \n",
                "    peak_map50 = df['metrics/mAP50(B)'].max() * 100\n",
                "    final_map50 = df['metrics/mAP50(B)'].iloc[-1] * 100\n",
                "    peak_map50_95 = df['metrics/mAP50-95(B)'].max() * 100\n",
                "    final_map50_95 = df['metrics/mAP50-95(B)'].iloc[-1] * 100\n",
                "    \n",
                "    print('\\n' + '=' * 80)\n",
                "    print('                    THESIS BENCHMARK COMPARISON')\n",
                "    print('=' * 80)\n",
                "    print(f'Paddy-YOLO-EMA + Focal Loss Peak  mAP@50  : {peak_map50:.2f}% (mAP50-95: {peak_map50_95:.2f}%)')\n",
                "    print(f'Paddy-YOLO-EMA + Focal Loss Final mAP@50  : {final_map50:.2f}% (mAP50-95: {final_map50_95:.2f}%)')\n",
                "    print('-' * 80)\n",
                "    print('Paper Published Baselines (Zheng et al., 2025 Table 8):')\n",
                "    print('  - YOLOv8-N Baseline       : 61.1% mAP@50 | 43.3% mAP@50-95 | 60.1% F1')\n",
                "    print('  - Baseline Paddy-YOLO     : 70.2% mAP@50 | 51.5% mAP@50-95 | 67.7% F1')\n",
                "    print('-' * 80)\n",
                "    print('Previous Ablation Runs on Kaggle:')\n",
                "    print('  - Replicated Baseline     : 69.41% mAP@50 (val) | 68.34% mAP@50 (test)')\n",
                "    print('  - 4-Head (P2 + EMA)       : 67.03% mAP@50 (val) | 50.38% mAP@50-95 (val)')\n",
                "    print('  - 3-Head (EMA Only)       : 65.20% mAP@50 (val) | 68.50% mAP@50 (test)')\n",
                "    print('=' * 80)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Package artifacts for download\n",
                "import shutil\n",
                "train_run_dir = '/kaggle/working/runs/train/paddy_yolo_focal_ema_fold0'\n",
                "zip_dest = '/kaggle/working/paddy_yolo_focal_ema_fold0_results'\n",
                "if os.path.exists(train_run_dir):\n",
                "    shutil.make_archive(zip_dest, 'zip', train_run_dir)\n",
                "    print(f'Created downloadable results archive at: {zip_dest}.zip')\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_file = out_dir / "paddy_yolo_focal_ema_training.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

# 3. Create kernel-metadata.json
metadata = {
    "id": "sednaaa90377/paddy-yolo-focal-ema-r2000",
    "title": "paddy-yolo-focal-ema-r2000",
    "code_file": "paddy_yolo_focal_ema_training.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [
        "dmrashidferrer/r2000v1-2024-images"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

meta_file = out_dir / "kernel-metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Paddy-YOLO Focal-EMA kernel files generated in kaggle_enhanced_kernel/ successfully.")
