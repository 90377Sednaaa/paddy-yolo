import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_fold0_ema_kernel")
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
                "kaggle_fold0_ema_kernel",
                "kaggle_5fold_scratch_kernel", "kaggle_5fold_baseline_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Fast Fold 0 Training of Paddy-YOLO-EMA
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Paddy-YOLO-EMA Fold 0 Fast Benchmark (3-Head + EMA Attention)\n",
                "\n",
                "This notebook executes a **targeted single-fold (Fold 0) benchmark** of **Paddy-YOLO-EMA** (sequential RGGConv + 3 EMA Attention Blocks + DIMIoU Strategy H) on the **R2000** rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Configuration:\n",
                "- **Architecture**: `configs/paddy-yolo-ema.yaml` (4 sequential RGG modules, 3 EMA attention blocks before P3, P4, P5 heads)\n",
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
                "print('Generated 5-fold cross validation split configs.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Train Paddy-YOLO-EMA Fold 0 (Unbuffered Live Streamed Output)\n",
                "# Using -u ensures every epoch line is flushed immediately to stdout & logs\n",
                "!python -u train.py \\\n",
                "    --model configs/paddy-yolo-ema.yaml \\\n",
                "    --weights yolov8n.pt \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --hyp configs/hyp_paper.yaml \\\n",
                "    --epochs 300 \\\n",
                "    --batch 32 \\\n",
                "    --imgsz 640 \\\n",
                "    --cache True \\\n",
                "    --device 0 \\\n",
                "    --project /kaggle/working/runs/train \\\n",
                "    --name paddy_yolo_ema_fold0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Evaluate Best Model on Unseen Fold 0 Test Split\n",
                "!python -u val.py \\\n",
                "    --weights /kaggle/working/runs/train/paddy_yolo_ema_fold0/weights/best.pt \\\n",
                "    --data /kaggle/working/data/r2000/data_fold0.yaml \\\n",
                "    --split test \\\n",
                "    --imgsz 640 \\\n",
                "    --device 0\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Detailed Synthesis & Comparison: EMA Fold 0 vs. Baseline Fold 0\n",
                "import numpy as np, pandas as pd\n",
                "from pathlib import Path\n",
                "from ultralytics import YOLO\n",
                "from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES\n",
                "\n",
                "print('=' * 85)\n",
                "print('SYNTHESIZING FOLD 0 RESULTS: PADDY-YOLO-EMA VS BASELINE PADDY-YOLO')\n",
                "print('=' * 85)\n",
                "\n",
                "best_ckpt = '/kaggle/working/runs/train/paddy_yolo_ema_fold0/weights/best.pt'\n",
                "data_yaml = '/kaggle/working/data/r2000/data_fold0.yaml'\n",
                "\n",
                "eval_model = YOLO(best_ckpt)\n",
                "metrics = eval_model.val(\n",
                "    data=data_yaml,\n",
                "    split='test',\n",
                "    imgsz=640,\n",
                "    batch=32,\n",
                "    device=0,\n",
                "    plots=True,\n",
                "    verbose=False\n",
                ")\n",
                "\n",
                "p_all = float(metrics.box.mp) * 100.0\n",
                "r_all = float(metrics.box.mr) * 100.0\n",
                "map50_all = float(metrics.box.map50) * 100.0\n",
                "map50_95_all = float(metrics.box.map) * 100.0\n",
                "f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-6)\n",
                "\n",
                "p_per_class = metrics.box.p * 100.0\n",
                "r_per_class = metrics.box.r * 100.0\n",
                "map50_per_class = metrics.box.ap50 * 100.0\n",
                "map50_95_per_class = metrics.box.ap * 100.0\n",
                "\n",
                "# Baseline Fold 0 overall metrics for comparison\n",
                "baseline_fold0_p = 70.71\n",
                "baseline_fold0_r = 61.82\n",
                "baseline_fold0_map50 = 67.55\n",
                "baseline_fold0_map50_95 = 47.75\n",
                "baseline_fold0_f1 = 65.97\n",
                "\n",
                "rows = []\n",
                "for idx, c_name in enumerate(R2000_CLASSES):\n",
                "    p_c = float(p_per_class[idx]) if idx < len(p_per_class) else 0.0\n",
                "    r_c = float(r_per_class[idx]) if idx < len(r_per_class) else 0.0\n",
                "    m50_c = float(map50_per_class[idx]) if idx < len(map50_per_class) else 0.0\n",
                "    m50_95_c = float(map50_95_per_class[idx]) if idx < len(map50_95_per_class) else 0.0\n",
                "    rows.append({\n",
                "        'Species': c_name,\n",
                "        'P (%)': f'{p_c:.1f}',\n",
                "        'R (%)': f'{r_c:.1f}',\n",
                "        'mAP@50 (%)': f'{m50_c:.1f}',\n",
                "        'mAP@50-95 (%)': f'{m50_95_c:.1f}'\n",
                "    })\n",
                "\n",
                "# Summary row\n",
                "rows.append({\n",
                "    'Species': '--- OVERALL FOLD 0 ---',\n",
                "    'P (%)': f'{p_all:.2f}',\n",
                "    'R (%)': f'{r_all:.2f}',\n",
                "    'mAP@50 (%)': f'{map50_all:.2f}',\n",
                "    'mAP@50-95 (%)': f'{map50_95_all:.2f}'\n",
                "})\n",
                "\n",
                "df_results = pd.DataFrame(rows)\n",
                "print('\\n--- PADDY-YOLO-EMA FOLD 0 SPECIES METRICS ---')\n",
                "print(df_results.to_string(index=False))\n",
                "\n",
                "comp_data = [{\n",
                "    'Metric': 'Precision (%)',\n",
                "    'Baseline Fold 0': f'{baseline_fold0_p:.2f}',\n",
                "    'EMA Fold 0': f'{p_all:.2f}',\n",
                "    'Delta': f'{p_all - baseline_fold0_p:+.2f}%'\n",
                "}, {\n",
                "    'Metric': 'Recall (%)',\n",
                "    'Baseline Fold 0': f'{baseline_fold0_r:.2f}',\n",
                "    'EMA Fold 0': f'{r_all:.2f}',\n",
                "    'Delta': f'{r_all - baseline_fold0_r:+.2f}%'\n",
                "}, {\n",
                "    'Metric': 'mAP@50 (%)',\n",
                "    'Baseline Fold 0': f'{baseline_fold0_map50:.2f}',\n",
                "    'EMA Fold 0': f'{map50_all:.2f}',\n",
                "    'Delta': f'{map50_all - baseline_fold0_map50:+.2f}%'\n",
                "}, {\n",
                "    'Metric': 'mAP@50-95 (%)',\n",
                "    'Baseline Fold 0': f'{baseline_fold0_map50_95:.2f}',\n",
                "    'EMA Fold 0': f'{map50_95_all:.2f}',\n",
                "    'Delta': f'{map50_95_all - baseline_fold0_map50_95:+.2f}%'\n",
                "}, {\n",
                "    'Metric': 'F1 Score (%)',\n",
                "    'Baseline Fold 0': f'{baseline_fold0_f1:.2f}',\n",
                "    'EMA Fold 0': f'{f1_all:.2f}',\n",
                "    'Delta': f'{f1_all - baseline_fold0_f1:+.2f}%'\n",
                "}]\n",
                "\n",
                "df_comp = pd.DataFrame(comp_data)\n",
                "print('\\n' + '=' * 60)\n",
                "print('HEAD-TO-HEAD COMPARISON: EMA FOLD 0 VS BASELINE FOLD 0')\n",
                "print('=' * 60)\n",
                "print(df_comp.to_string(index=False))\n",
                "\n",
                "# Export reports\n",
                "df_results.to_csv('/kaggle/working/paddy_yolo_ema_fold0_species_results.csv', index=False)\n",
                "df_comp.to_csv('/kaggle/working/paddy_yolo_ema_fold0_vs_baseline.csv', index=False)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8: Package weights and artifacts for download\n",
                "import shutil\n",
                "train_dir = '/kaggle/working/runs/train'\n",
                "zip_path = '/kaggle/working/paddy_yolo_ema_fold0_artifacts'\n",
                "if os.path.exists(train_dir):\n",
                "    shutil.make_archive(zip_path, 'zip', train_dir)\n",
                "    print(f'Archive created: {zip_path}.zip')\n"
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

nb_path = out_dir / "paddy_yolo_ema_fold0.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Jupyter Notebook generated at: {nb_path}")

meta = {
    "id": "leanadrianmurillo/paddy-yolo-ema-fold0",
    "title": "paddy-yolo-ema-fold0",
    "code_file": "paddy_yolo_ema_fold0.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [
        "dmrashidferrer/r2000v1-2024-images"
    ],
    "competition_sources": [],
    "kernel_sources": []
}

meta_path = out_dir / "kernel-metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"Kernel metadata generated at: {meta_path}")
