import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_5fold_dysample_kernel")
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
                "kaggle_5fold_dysample_kernel",
                "kaggle_5fold_scratch_kernel", "kaggle_5fold_baseline_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for 5-Fold Cross Validation of Paddy-YOLO with DySample
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌾 Master 5-Fold Cross-Validation: Paddy-YOLO with DySample (Configuration C)\n",
                "\n",
                "This notebook executes the **master 5-fold cross-validation benchmark** of the **Paddy-YOLO-DySample** model (sequential RGGConv + 2 DySample Dynamic Upsamplers in Neck + DIMIoU Strategy H) on the **R2000** rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Experimental Configuration:\n",
                "- **Architecture**: `configs/paddy-yolo-dysample.yaml` (4 sequential RGG modules in backbone, 2 DySample dynamic upsamplers in neck, 3 heads)\n",
                "- **Bounding Box Loss**: DIMIoU with Strategy H dynamic ratio scheduling\n",
                "- **Weight Initialization**: COCO backbone pre-training transfer (`yolov8n.pt`)\n",
                "- **Training Schedule**: 300 epochs, batch size 32, image size 640x640, AdamW optimizer\n",
                "- **Learning Rate**: `lr0=0.01`, `lrf=0.01` (constant schedule, Table 6)\n",
                "- **Evaluation**: 5-Fold Cross-Validation macro average (\\mu \\pm \\sigma) across all 16 species"
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
                "# Cell 3: Install Ultralytics and Pre-fetch yolov8n.pt\n",
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
                "# Cell 5: Sequential 5-Fold Training Loop (Folds 0, 1, 2, 3, 4)\n",
                "import time, os, subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "os.environ['PYTHONUNBUFFERED'] = '1'\n",
                "\n",
                "def run_command_live(cmd):\n",
                "    print(f'>>> Executing: {\" \".join(cmd)}', flush=True)\n",
                "    proc = subprocess.Popen(\n",
                "        cmd,\n",
                "        stdout=subprocess.PIPE,\n",
                "        stderr=subprocess.STDOUT,\n",
                "        text=True,\n",
                "        bufsize=1,\n",
                "        universal_newlines=True\n",
                "    )\n",
                "    for line in iter(proc.stdout.readline, ''):\n",
                "        if line:\n",
                "            print(line, end='', flush=True)\n",
                "    proc.stdout.close()\n",
                "    rc = proc.wait()\n",
                "    if rc != 0:\n",
                "        raise subprocess.CalledProcessError(rc, cmd)\n",
                "\n",
                "overall_start_time = time.time()\n",
                "completed_folds = []\n",
                "\n",
                "for fold in [0, 1, 2, 3, 4]:\n",
                "    fold_start = time.time()\n",
                "    elapsed_overall = (time.time() - overall_start_time) / 3600.0\n",
                "    print('=' * 85, flush=True)\n",
                "    print(f'STARTING FOLD {fold} / 4 (Overall Elapsed: {elapsed_overall:.2f} hrs)', flush=True)\n",
                "    print('=' * 85, flush=True)\n",
                "    \n",
                "    data_yaml = f'/kaggle/working/data/r2000/data_fold{fold}.yaml'\n",
                "    run_name = f'paddy_yolo_dysample_fold{fold}'\n",
                "    \n",
                "    # Train 300 epochs with COCO backbone transfer and RAM caching\n",
                "    cmd_train = [\n",
                "        'python', '-u', 'train.py',\n",
                "        '--model', 'configs/paddy-yolo-dysample.yaml',\n",
                "        '--weights', 'yolov8n.pt',\n",
                "        '--data', data_yaml,\n",
                "        '--hyp', 'configs/hyp_paper.yaml',\n",
                "        '--epochs', '300',\n",
                "        '--batch', '32',\n",
                "        '--imgsz', '640',\n",
                "        '--cache', 'True',\n",
                "        '--device', '0',\n",
                "        '--project', '/kaggle/working/runs/train',\n",
                "        '--name', run_name\n",
                "    ]\n",
                "    run_command_live(cmd_train)\n",
                "    \n",
                "    # Evaluate on unseen test split\n",
                "    best_ckpt = f'/kaggle/working/runs/train/{run_name}/weights/best.pt'\n",
                "    cmd_val = [\n",
                "        'python', '-u', 'val.py',\n",
                "        '--weights', best_ckpt,\n",
                "        '--data', data_yaml,\n",
                "        '--split', 'test',\n",
                "        '--imgsz', '640',\n",
                "        '--device', '0'\n",
                "    ]\n",
                "    run_command_live(cmd_val)\n",
                "    \n",
                "    fold_duration_m = (time.time() - fold_start) / 60.0\n",
                "    completed_folds.append(fold)\n",
                "    print(f'✅ Fold {fold} completed successfully in {fold_duration_m:.1f} minutes!', flush=True)\n",
                "\n",
                "total_time_h = (time.time() - overall_start_time) / 3600.0\n",
                "print(f'\\nAll 5 folds finished in {total_time_h:.2f} hours. Completed folds: {completed_folds}', flush=True)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Cross-Validation Synthesis & 3-Way Comparison Table (DySample vs Baseline vs Paper)\n",
                "import numpy as np, pandas as pd\n",
                "from pathlib import Path\n",
                "from ultralytics import YOLO\n",
                "from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES\n",
                "\n",
                "print('=' * 85)\n",
                "print('SYNTHESIZING 5-FOLD CROSS-VALIDATION RESULTS FOR PADDY-YOLO-DYSAMPLE')\n",
                "print('=' * 85)\n",
                "\n",
                "fold_records = []\n",
                "class_records = {c: {'p': [], 'r': [], 'map50': [], 'map50_95': []} for c in R2000_CLASSES}\n",
                "\n",
                "for fold in completed_folds:\n",
                "    best_ckpt = f'/kaggle/working/runs/train/paddy_yolo_dysample_fold{fold}/weights/best.pt'\n",
                "    data_yaml = f'/kaggle/working/data/r2000/data_fold{fold}.yaml'\n",
                "    \n",
                "    if not Path(best_ckpt).exists():\n",
                "        continue\n",
                "        \n",
                "    eval_model = YOLO(best_ckpt)\n",
                "    metrics = eval_model.val(\n",
                "        data=data_yaml,\n",
                "        split='test',\n",
                "        imgsz=640,\n",
                "        batch=32,\n",
                "        device=0,\n",
                "        plots=False,\n",
                "        verbose=False\n",
                "    )\n",
                "    \n",
                "    p_all = float(metrics.box.mp) * 100.0\n",
                "    r_all = float(metrics.box.mr) * 100.0\n",
                "    map50_all = float(metrics.box.map50) * 100.0\n",
                "    map50_95_all = float(metrics.box.map) * 100.0\n",
                "    f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-6)\n",
                "    \n",
                "    fold_records.append({\n",
                "        'Fold': fold,\n",
                "        'Precision (%)': p_all,\n",
                "        'Recall (%)': r_all,\n",
                "        'mAP@50 (%)': map50_all,\n",
                "        'mAP@50-95 (%)': map50_95_all,\n",
                "        'F1 Score (%)': f1_all\n",
                "    })\n",
                "    \n",
                "    p_per_class = metrics.box.p * 100.0\n",
                "    r_per_class = metrics.box.r * 100.0\n",
                "    map50_per_class = metrics.box.ap50 * 100.0\n",
                "    map50_95_per_class = metrics.box.ap * 100.0\n",
                "    \n",
                "    for idx, c_name in enumerate(R2000_CLASSES):\n",
                "        if idx < len(p_per_class):\n",
                "            class_records[c_name]['p'].append(float(p_per_class[idx]))\n",
                "            class_records[c_name]['r'].append(float(r_per_class[idx]))\n",
                "            class_records[c_name]['map50'].append(float(map50_per_class[idx]))\n",
                "            class_records[c_name]['map50_95'].append(float(map50_95_per_class[idx]))\n",
                "\n",
                "df_folds = pd.DataFrame(fold_records)\n",
                "print('\\n--- INDIVIDUAL FOLD TEST RESULTS (PADDY-YOLO-DYSAMPLE) ---')\n",
                "print(df_folds.to_string(index=False))\n",
                "\n",
                "# Baseline Recreated 5-Fold mAP@50 reference values\n",
                "baseline_map50 = {\n",
                "    'rice leaf roller': 60.3, 'pink rice borer': 74.0, 'asiatic rice borer': 27.4,\n",
                "    'yellow rice borer': 61.1, 'rice leaf caterpillar': 44.7, 'small brown plant hopper': 63.2,\n",
                "    'white backed plant hopper': 53.8, 'brown plant hopper': 57.7, 'rice leaf hopper': 72.8,\n",
                "    'rice gall midge': 72.1, 'paddy stem maggot': 70.0, 'rice stem fly': 83.1,\n",
                "    'grain spreader thrips': 70.7, 'rice weevil': 66.4, 'rice shell pest': 78.4,\n",
                "    'melanitis leda': 81.2\n",
                "}\n",
                "\n",
                "# Paper Table 8 published values\n",
                "paper_map50 = {\n",
                "    'rice leaf roller': 66.5, 'pink rice borer': 77.2, 'asiatic rice borer': 33.8,\n",
                "    'yellow rice borer': 65.1, 'rice leaf caterpillar': 52.7, 'small brown plant hopper': 74.8,\n",
                "    'white backed plant hopper': 57.9, 'brown plant hopper': 63.9, 'rice leaf hopper': 79.4,\n",
                "    'rice gall midge': 82.4, 'paddy stem maggot': 81.2, 'rice stem fly': 85.6,\n",
                "    'grain spreader thrips': 74.6, 'rice weevil': 68.1, 'rice shell pest': 79.1,\n",
                "    'melanitis leda': 80.7\n",
                "}\n",
                "\n",
                "synthesis_rows = []\n",
                "for c_name in R2000_CLASSES:\n",
                "    vals = class_records[c_name]\n",
                "    m50_mean = np.mean(vals['map50']) if vals['map50'] else 0.0\n",
                "    m50_std = np.std(vals['map50']) if vals['map50'] else 0.0\n",
                "    p_mean = np.mean(vals['p']) if vals['p'] else 0.0\n",
                "    r_mean = np.mean(vals['r']) if vals['r'] else 0.0\n",
                "    m50_95_mean = np.mean(vals['map50_95']) if vals['map50_95'] else 0.0\n",
                "    base_m50 = baseline_map50.get(c_name, 0.0)\n",
                "    paper_m50_val = paper_map50.get(c_name, 0.0)\n",
                "    delta_base = m50_mean - base_m50\n",
                "    delta_paper = m50_mean - paper_m50_val\n",
                "    synthesis_rows.append({\n",
                "        'Species': c_name,\n",
                "        'P (%)': f'{p_mean:.1f}',\n",
                "        'R (%)': f'{r_mean:.1f}',\n",
                "        'DySample mAP@50 (μ ± σ)': f'{m50_mean:.1f} ± {m50_std:.1f}',\n",
                "        'Baseline mAP@50': f'{base_m50:.1f}',\n",
                "        'Δ vs Baseline': f'{delta_base:+.1f}%',\n",
                "        'Paper mAP@50': f'{paper_m50_val:.1f}',\n",
                "        'Δ vs Paper': f'{delta_paper:+.1f}%'\n",
                "    })\n",
                "\n",
                "# Macro row\n",
                "mean_p = df_folds['Precision (%)'].mean()\n",
                "std_p = df_folds['Precision (%)'].std()\n",
                "mean_r = df_folds['Recall (%)'].mean()\n",
                "std_r = df_folds['Recall (%)'].std()\n",
                "mean_map50 = df_folds['mAP@50 (%)'].mean()\n",
                "std_map50 = df_folds['mAP@50 (%)'].std()\n",
                "mean_map50_95 = df_folds['mAP@50-95 (%)'].mean()\n",
                "std_map50_95 = df_folds['mAP@50-95 (%)'].std()\n",
                "\n",
                "synthesis_rows.append({\n",
                "    'Species': '--- OVERALL (MACRO 5-FOLD) ---',\n",
                "    'P (%)': f'{mean_p:.1f} ± {std_p:.1f}',\n",
                "    'R (%)': f'{mean_r:.1f} ± {std_r:.1f}',\n",
                "    'DySample mAP@50 (μ ± σ)': f'{mean_map50:.1f} ± {std_map50:.1f}',\n",
                "    'Baseline mAP@50': '64.8%',\n",
                "    'Δ vs Baseline': f'{mean_map50 - 64.8:+.1f}%',\n",
                "    'Paper mAP@50': '70.2%',\n",
                "    'Δ vs Paper': f'{mean_map50 - 70.2:+.1f}%'\n",
                "})\n",
                "\n",
                "df_synthesis = pd.DataFrame(synthesis_rows)\n",
                "print('\\n--- 5-FOLD MACRO SUMMARY: DYSAMPLE VS BASELINE VS PAPER ---')\n",
                "print(df_synthesis.to_string(index=False))\n",
                "\n",
                "# Export CSV reports\n",
                "df_folds.to_csv('/kaggle/working/paddy_yolo_dysample_individual_folds.csv', index=False)\n",
                "df_synthesis.to_csv('/kaggle/working/paddy_yolo_dysample_5fold_master_summary.csv', index=False)\n"
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

nb_path = out_dir / "paddy_yolo_dysample_5fold.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook saved to: {nb_path} ({nb_path.stat().st_size} bytes)")

# 3. Kernel metadata
meta = {
    "id": "leanadrianmurillo/paddy-yolo-dysample-5fold",
    "title": "paddy-yolo-dysample-5fold",
    "code_file": "paddy_yolo_dysample_5fold.ipynb",
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
print("DySample 5-Fold Cross Validation Kaggle Kernel Builder completed successfully!")
