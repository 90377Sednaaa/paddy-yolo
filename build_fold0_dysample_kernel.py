import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_fold0_dysample_kernel")
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
                "kaggle_fold0_ema_kernel", "kaggle_fold0_dysample_kernel",
                "kaggle_5fold_scratch_kernel", "kaggle_5fold_baseline_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Fast Fold 0 Training of Paddy-YOLO with DySample
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Paddy-YOLO with DySample (Configuration C) Fold 0 Fast Benchmark\n",
                "\n",
                "This notebook executes a **targeted single-fold (Fold 0) benchmark** of **Paddy-YOLO-DySample** (sequential RGGConv + DySample Dynamic Upsampling + DIMIoU Strategy H) on the **R2000** rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Configuration:\n",
                "- **Architecture**: `configs/paddy-yolo-dysample.yaml` (4 sequential RGG modules in backbone, 2 DySample dynamic upsamplers in neck)\n",
                "- **Loss**: Dynamic Inner-MPDIoU (DIMIoU Strategy H)\n",
                "- **Initialization**: Transfer learning from COCO (`yolov8n.pt`)\n",
                "- **Split**: Fold 0 (Train: 3 folds, Val: 1 fold, Test: 1 fold)\n",
                "- **Hyperparameters**: 300 epochs, batch size 32, imgsz 640, AdamW (`configs/hyp_paper.yaml`)\n",
                "- **Comparison**: Direct side-by-side comparison against Baseline Paddy-YOLO Fold 0 (67.55% mAP@50)"
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
                "# Cell 3: Install Ultralytics and Run Smoke Test\n",
                "!pip install ultralytics -q\n",
                "!python verify_paddy_yolo.py --smoke-test\n",
                "\n",
                "from ultralytics import YOLO\n",
                "base_model = YOLO('yolov8n.pt')\n",
                "print('Base yolov8n.pt successfully loaded.')\n"
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
                "# Cell 5: Train Fold 0 of Paddy-YOLO with DySample (300 epochs, AdamW)\n",
                "import time, os, subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "start_time = time.time()\n",
                "data_yaml = '/kaggle/working/data/r2000/data_fold0.yaml'\n",
                "run_name = 'paddy_yolo_dysample_fold0'\n",
                "\n",
                "cmd_train = [\n",
                "    'python', '-u', 'train.py',\n",
                "    '--model', 'configs/paddy-yolo-dysample.yaml',\n",
                "    '--weights', 'yolov8n.pt',\n",
                "    '--data', data_yaml,\n",
                "    '--hyp', 'configs/hyp_paper.yaml',\n",
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
                "print(f'Training completed in {elapsed_train:.2f} hours.')\n",
                "\n",
                "# Evaluate on Fold 0 unseen test set\n",
                "best_ckpt = f'/kaggle/working/runs/train/{run_name}/weights/best.pt'\n",
                "cmd_val = [\n",
                "    'python', '-u', 'val.py',\n",
                "    '--weights', best_ckpt,\n",
                "    '--data', data_yaml,\n",
                "    '--split', 'test',\n",
                "    '--imgsz', '640',\n",
                "    '--device', '0'\n",
                "]\n",
                "print(f'Executing Test Evaluation: {\" \".join(cmd_val)}')\n",
                "subprocess.run(cmd_val, check=True)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Extract & Synthesize Fold 0 Test Performance vs Baseline\n",
                "import pandas as pd, numpy as np\n",
                "from ultralytics import YOLO\n",
                "from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES\n",
                "\n",
                "best_ckpt = '/kaggle/working/runs/train/paddy_yolo_dysample_fold0/weights/best.pt'\n",
                "data_yaml = '/kaggle/working/data/r2000/data_fold0.yaml'\n",
                "\n",
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
                "# Baseline Fold 0 published reference values\n",
                "base_p = 68.32\n",
                "base_r = 59.20\n",
                "base_map50 = 67.55\n",
                "base_map50_95 = 48.71\n",
                "base_f1 = 63.43\n",
                "\n",
                "summary_data = [\n",
                "    {\n",
                "        'Metric': 'Precision (%)',\n",
                "        'Baseline Fold 0': f'{base_p:.2f}%',\n",
                "        'DySample Fold 0': f'{p_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{p_all - base_p:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'Recall (%)',\n",
                "        'Baseline Fold 0': f'{base_r:.2f}%',\n",
                "        'DySample Fold 0': f'{r_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{r_all - base_r:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50 (%)',\n",
                "        'Baseline Fold 0': f'{base_map50:.2f}%',\n",
                "        'DySample Fold 0': f'{map50_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{map50_all - base_map50:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50-95 (%)',\n",
                "        'Baseline Fold 0': f'{base_map50_95:.2f}%',\n",
                "        'DySample Fold 0': f'{map50_95_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{map50_95_all - base_map50_95:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'F1 Score (%)',\n",
                "        'Baseline Fold 0': f'{base_f1:.2f}%',\n",
                "        'DySample Fold 0': f'{f1_all:.2f}%',\n",
                "        'Delta vs Baseline': f'{f1_all - base_f1:+.2f}%'\n",
                "    }\n",
                "]\n",
                "\n",
                "df_summary = pd.DataFrame(summary_data)\n",
                "print('\\n' + '=' * 80)\n",
                "print('PADDY-YOLO DYSAMPLE FOLD 0 TEST BENCHMARK VS BASELINE')\n",
                "print('=' * 80)\n",
                "print(df_summary.to_string(index=False))\n",
                "df_summary.to_csv('paddy_yolo_dysample_fold0_vs_baseline.csv', index=False)\n",
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
                "print('\\n' + '=' * 80)\n",
                "print('PER-SPECIES PERFORMANCE (DYSAMPLE FOLD 0)')\n",
                "print('=' * 80)\n",
                "print(df_species.to_string(index=False))\n",
                "df_species.to_csv('paddy_yolo_dysample_fold0_species_results.csv', index=False)\n"
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

nb_path = out_dir / "paddy_yolo_dysample_fold0.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook saved to: {nb_path} ({nb_path.stat().st_size} bytes)")

# 3. Kernel metadata
meta = {
    "id": "leanadrianmurillo/paddy-yolo-dysample-fold0",
    "title": "paddy-yolo-dysample-fold0",
    "code_file": "paddy_yolo_dysample_fold0.ipynb",
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
print("DySample Fold 0 Kaggle Kernel Builder completed successfully!")
