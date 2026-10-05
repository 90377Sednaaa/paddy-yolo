import io
import zipfile
import os
import base64
import json
from pathlib import Path

# Create kaggle_kernel directory
out_dir = Path("kaggle_kernel")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Zip local codebase
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".pytest_cache", ".superpowers", "docs", "runs", "data", "kaggle_kernel", ".vscode", ".idea"]]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created. Size: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Paddy-YOLO: Baseline YOLOv8-N Training on R2000\n",
                "\n",
                "This notebook trains the **Baseline YOLOv8-N** model on the **R2000** rice pest dataset (`dmrashidferrer/r2000v1-2024-images`) to reproduce the baseline benchmark from Zheng et al. (*Computers and Electronics in Agriculture*, 2025).\n",
                "\n",
                "### Configuration:\n",
                "- **Model Architecture**: YOLOv8-N (standard)\n",
                "- **Bounding Box Loss**: CIoU\n",
                "- **Optimizer**: AdamW, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005\n",
                "- **Training Schedule**: 300 epochs, batch size 32, image size 640x640\n",
                "- **Split**: 5-Fold Cross Validation Fold 0 (3 folds train, 1 fold val, 1 fold test)"
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
                "# Cell 5: Train Baseline YOLOv8-N (300 Epochs, AdamW, CIoU)\n",
                "!python train.py \\\n",
                "    --baseline \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --hyp configs/hyp_paper.yaml \\\n",
                "    --epochs 300 \\\n",
                "    --batch 32 \\\n",
                "    --imgsz 640 \\\n",
                "    --device 0 \\\n",
                "    --project /kaggle/working/runs/train \\\n",
                "    --name baseline_yolov8n_fold0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Evaluate Baseline Checkpoint on Test Set\n",
                "best_weights = '/kaggle/working/runs/train/baseline_yolov8n_fold0/weights/best.pt'\n",
                "\n",
                "!python val.py \\\n",
                "    --weights {best_weights} \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --split test \\\n",
                "    --device 0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Package artifacts for download\n",
                "import shutil\n",
                "train_run_dir = '/kaggle/working/runs/train/baseline_yolov8n_fold0'\n",
                "zip_dest = '/kaggle/working/baseline_yolov8n_fold0_results'\n",
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

nb_file = out_dir / "paddy_yolo_baseline_training.ipynb"
with open(nb_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

# 3. Create kernel-metadata.json
metadata = {
    "id": "sednaaa90377/paddy-yolo-baseline-yolov8-n-r2000",
    "title": "paddy-yolo-baseline-yolov8-n-r2000",
    "code_file": "paddy_yolo_baseline_training.ipynb",
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

print("Kaggle kernel files regenerated in kaggle_kernel/ successfully.")
