import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_ema_800_kernel")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Zip local codebase
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if d not in [
                ".git", "__pycache__", ".pytest_cache", ".superpowers",
                "docs", "runs", "data", "kaggle_kernel", "kaggle_paddy_kernel",
                "kaggle_enhanced_kernel", "kaggle_ema_800_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created. Size: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Paddy-YOLO-EMA-800
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Enhanced Paddy-YOLO: Multi-Scale Attention (EMA) + High-Resolution (800px) Training\n",
                "\n",
                "This notebook trains and benchmarks the enhanced **Paddy-YOLO-EMA** model at **800x800 resolution** on the **R2000** rice pest dataset (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Scientific Motivation & Gap Closures (No Architectural Bias):\n",
                "1. **Core Architectural Innovation**: 3 Efficient Multi-Scale Attention (EMA, ICASSP 2023) modules placed immediately before detection heads in the neck to focus on pest textures while suppressing foliage background.\n",
                "2. **Pillar 1 - Transfer Learning**: Initialized with COCO pretrained backbone weights (`yolov8n.pt`) to provide mature visual filters (edges, textures, contours) rather than training from random Gaussian noise on 2,000 images.\n",
                "3. **Pillar 2 - High Resolution (800px)**: Increases pixel area of small pests (thrips, plant hoppers, gall midge) by **+56%** ($[800/640]^2 = 1.56$), enabling the 1D directional attention pooling in EMA to resolve fine antennae and legs.\n",
                "4. **Pillar 3 - Balanced Feature Augmentation**: MixUp (0.15) and Copy-Paste (0.15) synthesize rare camouflaged pests (stem borers, caterpillars) across diverse rice foliage backgrounds without altering loss gradient magnitudes.\n",
                "5. **Stable Loss Formulation**: Dynamic Inner-MPDIoU (DIMIoU Strategy H) + Standard BCE (Focal Loss disabled) ensuring maximum gradient flow for foreground class discrimination.\n",
                "6. **Edge Envelope**: 3.35M parameters (< 3.5M budget, >180 FPS on edge accelerators)."
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
                "# Cell 3: Install Ultralytics and Run Sanity Verification\n",
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
                "# Cell 5: Train Paddy-YOLO-EMA (300 Epochs, 800px, Pretrained Backbone, Balanced Augmentation, DIMIoU Strategy H)\n",
                "!python train.py \\\n",
                "    --model configs/paddy-yolo-ema.yaml \\\n",
                "    --weights yolov8n.pt \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --hyp configs/hyp_paper.yaml \\\n",
                "    --epochs 300 \\\n",
                "    --batch 16 \\\n",
                "    --imgsz 800 \\\n",
                "    --mixup 0.15 \\\n",
                "    --copy-paste 0.15 \\\n",
                "    --strategy H \\\n",
                "    --focal-gamma 0.0 \\\n",
                "    --device 0 \\\n",
                "    --project /kaggle/working/runs/train \\\n",
                "    --name paddy_yolo_ema_800_fold0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Rigorous Multi-Stage Evaluation on Validation & Test Sets\n",
                "import os\n",
                "best_weights = '/kaggle/working/runs/train/paddy_yolo_ema_800_fold0/weights/best.pt'\n",
                "\n",
                "print('=' * 80)\n",
                "print('STAGE 1: VALIDATION SET EVALUATION (Fold 0 Val @ 800px)')\n",
                "print('=' * 80)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split val --imgsz 800 --device 0\n",
                "\n",
                "print('\\n' + '=' * 80)\n",
                "print('STAGE 2: UNSEEN TEST SET EVALUATION (Fold 0 Test @ 800px)')\n",
                "print('=' * 80)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split test --imgsz 800 --device 0\n",
                "\n",
                "print('\\n' + '=' * 80)\n",
                "print('STAGE 3: UNSEEN TEST SET WITH TEST-TIME AUGMENTATION (Fold 0 Test TTA @ 800px)')\n",
                "print('=' * 80)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split test --imgsz 800 --augment --device 0\n"
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
                "csv_path = Path('/kaggle/working/runs/train/paddy_yolo_ema_800_fold0/results.csv')\n",
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
                "    print(f'Paddy-YOLO-EMA (800px) Peak  mAP@50    : {peak_map50:.2f}% (mAP50-95: {peak_map50_95:.2f}%)')\n",
                "    print(f'Paddy-YOLO-EMA (800px) Final mAP@50    : {final_map50:.2f}% (mAP50-95: {final_map50_95:.2f}%)')\n",
                "    print('-' * 80)\n",
                "    print('Paper Published Baselines (Zheng et al., 2025 Table 8):')\n",
                "    print('  - YOLOv8-N Baseline       : 61.1% mAP@50 | 43.3% mAP@50-95 | 60.1% F1')\n",
                "    print('  - Baseline Paddy-YOLO     : 70.2% mAP@50 | 51.5% mAP@50-95 | 67.7% F1')\n",
                "    print('-' * 80)\n",
                "    print('Previous Kaggle Runs on Fold 0:')\n",
                "    print('  - Replicated YOLOv8-N     : 59.81% mAP@50 (val) | 59.98% mAP@50 (test)')\n",
                "    print('  - Replicated Paddy-YOLO   : 69.41% mAP@50 (val) | 68.34% mAP@50 (test)')\n",
                "    print('  - Run 3 (Paddy-YOLO-EMA 640px) : 65.20% mAP@50 (val) | 68.50% mAP@50 (test)')\n",
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
                "train_run_dir = '/kaggle/working/runs/train/paddy_yolo_ema_800_fold0'\n",
                "zip_dest = '/kaggle/working/paddy_yolo_ema_800_fold0_results'\n",
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

nb_file = out_dir / "paddy_yolo_ema_800_training.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

# 3. Create kernel-metadata.json
metadata = {
    "id": "sednaaa90377/paddy-yolo-ema-800-r2000",
    "title": "paddy-yolo-ema-800-r2000",
    "code_file": "paddy_yolo_ema_800_training.ipynb",
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

print("Paddy-YOLO-EMA-800 kernel files generated in kaggle_ema_800_kernel/ successfully.")
