import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_paddy_kernel")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Zip local codebase
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in [".git", ".venv", "__pycache__", ".pytest_cache", ".superpowers", "docs", "runs", "data", "kaggle_kernel", "kaggle_paddy_kernel", ".vscode", ".idea"]]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created. Size: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for actual Paddy-YOLO training
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Paddy-YOLO: Proposed Architecture Training on R2000\n",
                "\n",
                "This notebook trains the actual **Paddy-YOLO** model (RGG Backbone + Dynamic Inner-MPDIoU / DIMIoU Loss) on the **R2000** rice pest dataset (`dmrashidpferrer/r2000-pestnet`).\n",
                "\n",
                "### Architectural Innovations (Zheng et al., 2025):\n",
                "- **Backbone**: 4 RGG blocks replacing C2f in YOLOv8-N (RepConv + dual GhostConv with 5x5 DWConv + feature concatenation).\n",
                "- **Bounding Box Loss**: Dynamic Inner-MPDIoU (**DIMIoU**, Strategy H cosine scheduling).\n",
                "- **Optimizer**: AdamW, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, 300 epochs.\n",
                "- **Evaluation**: 5-Fold Cross Validation Fold 0 (3 train / 1 val / 1 test)."
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
                "# Cell 3: Install Ultralytics and Run Sanity Check\n",
                "!pip install ultralytics==8.3.253 -q\n",
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
                "            '-d', 'dmrashidpferrer/r2000-pestnet',\n",
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
                "# Cell 5: Train Actual Paddy-YOLO (RGG Backbone + Pretrained Backbone Init + DIMIoU Strategy H, 300 Epochs)\n",
                "!python train.py \\\n",
                "    --model configs/paddy-yolo.yaml \\\n",
                "    --weights yolov8n.pt \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --hyp configs/hyp_paper.yaml \\\n",
                "    --epochs 300 \\\n",
                "    --batch 32 \\\n",
                "    --imgsz 640 \\\n",
                "    --strategy H \\\n",
                "    --device 0 \\\n",
                "    --project /kaggle/working/runs/train \\\n",
                "    --name paddy_yolo_fold0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Evaluate Paddy-YOLO Checkpoint on Validation & Test Sets\n",
                "import os\n",
                "best_weights = '/kaggle/working/runs/train/paddy_yolo_fold0/weights/best.pt'\n",
                "\n",
                "print('=' * 60)\n",
                "print('EVALUATION ON VALIDATION SPLIT (Fold 0 Val)')\n",
                "print('=' * 60)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split val --device 0\n",
                "\n",
                "print('\\n' + '=' * 60)\n",
                "print('EVALUATION ON TEST SPLIT (Fold 0 Test)')\n",
                "print('=' * 60)\n",
                "!python val.py --weights {best_weights} --data /kaggle/working/data/r2000/data_fold0.yaml --split test --device 0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Benchmark Comparison against Paper Table 8\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "csv_path = Path('/kaggle/working/runs/train/paddy_yolo_fold0/results.csv')\n",
                "if csv_path.exists():\n",
                "    df = pd.read_csv(csv_path)\n",
                "    df.columns = df.columns.str.strip()\n",
                "    print('Training completed. Final 5 epochs metrics:')\n",
                "    cols = [c for c in ['epoch', 'train/box_loss', 'val/box_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'] if c in df.columns]\n",
                "    print(df[cols].tail(5).to_string())\n",
                "    \n",
                "    peak_map50 = df['metrics/mAP50(B)'].max() * 100\n",
                "    final_map50 = df['metrics/mAP50(B)'].iloc[-1] * 100\n",
                "    print('\\n' + '=' * 70)\n",
                "    print(f'Paddy-YOLO Peak mAP@50  : {peak_map50:.2f}%')\n",
                "    print(f'Paddy-YOLO Final mAP@50 : {final_map50:.2f}%')\n",
                "    print('Paper Reported Baseline : 61.1% (YOLOv8-N) | 70.2% (Paddy-YOLO Table 8)')\n",
                "    print('Thesis Requirement      : Within 2-3% of 70.2% (Target: >= 67.2% - 70.2%)')\n",
                "    print('=' * 70)\n"
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
                "train_run_dir = '/kaggle/working/runs/train/paddy_yolo_fold0'\n",
                "zip_dest = '/kaggle/working/paddy_yolo_fold0_results'\n",
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

nb_file = out_dir / "paddy_yolo_training.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

# 3. Create kernel-metadata.json
metadata = {
    "id": "sednaaa90377/paddy-yolo-actual-training-r2000",
    "title": "paddy-yolo-actual-training-r2000",
    "code_file": "paddy_yolo_training.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [
        "dmrashidpferrer/r2000-pestnet"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

meta_file = out_dir / "kernel-metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Paddy-YOLO kernel files generated in kaggle_paddy_kernel/ successfully.")
