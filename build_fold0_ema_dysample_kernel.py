import io
import zipfile
import os
import base64
import json
from pathlib import Path

out_dir = Path("kaggle_fold0_ema_dysample_kernel")
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
                "kaggle_5fold_dysample_kernel", "kaggle_fold0_ema_dysample_kernel",
                "kaggle_5fold_scratch_kernel", "kaggle_5fold_baseline_kernel", ".vscode", ".idea"
            ]
        ]
        for f in files:
            if f.endswith((".py", ".yaml", ".txt", ".md", ".ini")) and not f.endswith("review-pkg.diff"):
                p = os.path.join(root, f)
                z.write(p, p)

b64_code = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"Codebase archive created: {len(buf.getvalue())} bytes ({len(b64_code)} b64 chars).")

# 2. Build Jupyter Notebook for Fast Fold 0 Training of Paddy-YOLO with EMA + DySample (Configuration D)
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# \ud83c\udf3e Paddy-YOLO with EMA + DySample (Configuration D) Fold 0 Benchmark\n",
                "\n",
                "This notebook executes a **targeted single-fold (Fold 0) benchmark** of **Paddy-YOLO-EMA-DySample** (sequential RGGConv + EMA Multi-Scale Attention in Backbone + DySample Dynamic Upsamplers in Neck + DIMIoU Strategy H) on the **R2000** rice pest benchmark (`dmrashidferrer/r2000v1-2024-images`).\n",
                "\n",
                "### Configuration D Specifications:\n",
                "- **Backbone**: 4 sequential RGG modules + EMA multi-scale cross-spatial attention at P3, P4, P5 (Layers 5, 8, 11)\n",
                "- **Neck**: 2 DySample dynamic upsamplers (scale=2, style='lp', groups=4) replacing conventional nn.Upsample (Layers 13, 16)\n",
                "- **Bounding Box Loss**: Dynamic Inner-MPDIoU (DIMIoU Strategy H)\n",
                "- **Initialization**: Topological transfer learning from COCO (`yolov8n.pt`)\n",
                "- **Split**: Fold 0 (Train: 3 folds [1,214 imgs], Val: 1 fold [405 imgs], Test: 1 fold [405 imgs])\n",
                "- **Hyperparameters**: 300 epochs, batch size 32, imgsz 640, AdamW (`configs/hyp_paper.yaml`)\n",
                "- **Ablation Comparison**: Direct comparison against Recreated Baseline Fold 0 (67.55%), EMA Fold 0 (69.50%), and DySample Fold 0 (68.50%)"
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
                "from paddy_yolo.modules.register import register_paddy_modules\n",
                "from paddy_yolo.modules.ema import EMA\n",
                "from paddy_yolo.modules.dysample import DySample\n",
                "register_paddy_modules()\n",
                "\n",
                "model = YOLO('configs/paddy-yolo-ema-dysample.yaml')\n",
                "layers = list(model.model.model.children())\n",
                "print('Model successfully loaded! Total layers:', len(layers))\n",
                "print('Layer 5 (P3 EMA):', type(layers[5]).__name__)\n",
                "print('Layer 8 (P4 EMA):', type(layers[8]).__name__)\n",
                "print('Layer 11 (P5 EMA):', type(layers[11]).__name__)\n",
                "print('Layer 13 (DySample Up 1):', type(layers[13]).__name__)\n",
                "print('Layer 16 (DySample Up 2):', type(layers[16]).__name__)\n",
                "print('Layer 25 (Detect Head):', type(layers[25]).__name__, 'nc =', layers[25].nc)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4: Download COCO weights & Initialize Checkpoint with Topological Weight Transfer\n",
                "import urllib.request\n",
                "from pathlib import Path\n",
                "import torch\n",
                "from ultralytics import YOLO\n",
                "from paddy_yolo.modules.register import register_paddy_modules\n",
                "\n",
                "register_paddy_modules()\n",
                "weights_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'\n",
                "urllib.request.urlretrieve(weights_url, 'yolov8n.pt')\n",
                "print('Downloaded baseline yolov8n.pt successfully.')\n",
                "\n",
                "# Instantiate target architecture\n",
                "init_yolo = YOLO('configs/paddy-yolo-ema-dysample.yaml')\n",
                "init_yolo.load('yolov8n.pt')\n",
                "\n",
                "# Topological mapping for shifted backbone and neck layers\n",
                "ema_shift_map = {\n",
                "    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10, 9: 12,\n",
                "    12: 15, 15: 18, 16: 19, 18: 21, 19: 22, 21: 24, 22: 25,\n",
                "}\n",
                "src_ckpt = torch.load('yolov8n.pt', map_location='cpu', weights_only=False)\n",
                "src_sd = src_ckpt['model'].state_dict() if isinstance(src_ckpt, dict) and 'model' in src_ckpt else src_ckpt.state_dict()\n",
                "tgt_sd = init_yolo.model.state_dict()\n",
                "shifted_count = 0\n",
                "\n",
                "for src_key, src_val in src_sd.items():\n",
                "    if not src_key.startswith('model.'):\n",
                "        continue\n",
                "    parts = src_key.split('.')\n",
                "    src_idx = int(parts[1])\n",
                "    if src_idx in ema_shift_map:\n",
                "        tgt_key = f'model.{ema_shift_map[src_idx]}.' + '.'.join(parts[2:])\n",
                "        if tgt_key in tgt_sd and tgt_sd[tgt_key].shape == src_val.shape:\n",
                "            tgt_sd[tgt_key].copy_(src_val.float())\n",
                "            shifted_count += 1\n",
                "\n",
                "init_yolo.model.load_state_dict(tgt_sd)\n",
                "print(f'Topological weight transfer: transferred {shifted_count} tensor weights into shifted layers.')\n",
                "\n",
                "weights_dir = Path('weights')\n",
                "weights_dir.mkdir(parents=True, exist_ok=True)\n",
                "init_path = weights_dir / 'paddy-yolo-ema-dysample_init.pt'\n",
                "torch.save({'model': init_yolo.model}, init_path)\n",
                "print(f'Pre-trained initialized model saved at: {init_path}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5: Dataset Verification & Fold 0 Preparation\n",
                "from pathlib import Path\n",
                "import os\n",
                "from paddy_yolo.dataset.prepare_r2000 import prepare_kaggle_r2000, create_5fold_splits\n",
                "\n",
                "# Identify Kaggle dataset path\n",
                "candidate_dirs = [\n",
                "    Path('/kaggle/input/datasets/dmrashidferrer/r2000v1-2024-images'),\n",
                "    Path('/kaggle/input/r2000v1-2024-images'),\n",
                "    Path('/kaggle/input/r2000v1-2024-images/r2000v1-2024-images')\n",
                "]\n",
                "kaggle_r2000 = None\n",
                "for cd in candidate_dirs:\n",
                "    if cd.is_dir():\n",
                "        kaggle_r2000 = cd\n",
                "        break\n",
                "\n",
                "if kaggle_r2000 is None:\n",
                "    print('Listing /kaggle/input:')\n",
                "    for r, d, f in os.walk('/kaggle/input'):\n",
                "        print(r)\n",
                "        break\n",
                "    raise FileNotFoundError('R2000 Kaggle dataset directory not found!')\n",
                "\n",
                "print(f'Found Kaggle dataset at: {kaggle_r2000}')\n",
                "\n",
                "# Standardize Kaggle R2000 images & labels\n",
                "output_dir = Path('data/r2000')\n",
                "prepare_kaggle_r2000(input_dir=kaggle_r2000, output_dir=output_dir)\n",
                "\n",
                "# Generate official 5-fold cross-validation splits (seed=42, 60/20/20)\n",
                "folds = create_5fold_splits(dataset_dir=output_dir, seed=42)\n",
                "print('Generated 5-fold cross validation split configs:')\n",
                "for f in folds:\n",
                "    print(f'  -> {f}')\n",
                "fold0_yaml = output_dir / 'data_fold0.yaml'\n",
                "print(f'Fold 0 dataset config verified at: {fold0_yaml}')\n",
                "with open(fold0_yaml, 'r') as f:\n",
                "    print(f.read())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6: Train Paddy-YOLO with EMA + DySample (Fold 0, 300 epochs) with Live Unbuffered Logging\n",
                "import subprocess, sys\n",
                "\n",
                "cmd = [\n",
                "    sys.executable, '-u', 'train.py',\n",
                "    '--model', 'configs/paddy-yolo-ema-dysample.yaml',\n",
                "    '--weights', 'weights/paddy-yolo-ema-dysample_init.pt',\n",
                "    '--data', 'data/r2000/data_fold0.yaml',\n",
                "    '--epochs', '300',\n",
                "    '--batch', '32',\n",
                "    '--imgsz', '640',\n",
                "    '--hyp', 'configs/hyp_paper.yaml',\n",
                "    '--strategy', 'H',\n",
                "    '--device', '0',\n",
                "    '--name', 'paddy_yolo_ema_dysample_fold0',\n",
                "    '--project', 'runs/train'\n",
                "]\n",
                "\n",
                "print('Executing command: ' + ' '.join(cmd))\n",
                "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "for line in proc.stdout:\n",
                "    print(line, end='', flush=True)\n",
                "proc.wait()\n",
                "if proc.returncode != 0:\n",
                "    raise RuntimeError(f'Training failed with returncode {proc.returncode}')\n",
                "print('\\n>>> Fold 0 Training Finished Successfully! <<<\\n')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7: Evaluate Test Set on Fold 0 Best Checkpoint & Compare with All Ablations\n",
                "from ultralytics import YOLO\n",
                "import pandas as pd\n",
                "from paddy_yolo.modules.register import register_paddy_modules\n",
                "from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES\n",
                "from pathlib import Path\n",
                "\n",
                "register_paddy_modules()\n",
                "# Robustly search for best checkpoint in working directory\n",
                "best_candidates = list(Path('/kaggle/working').rglob('**/weights/best.pt'))\n",
                "if not best_candidates:\n",
                "    best_candidates = list(Path('.').rglob('**/weights/best.pt'))\n",
                "if not best_candidates:\n",
                "    raise FileNotFoundError('Best weights (best.pt) not found anywhere in working directory!')\n",
                "best_weights = best_candidates[0]\n",
                "print(f'Found best checkpoint for evaluation: {best_weights}')\n",
                "\n",
                "print(f'Loading best checkpoint for evaluation: {best_weights}')\n",
                "model = YOLO(str(best_weights))\n",
                "\n",
                "metrics = model.val(\n",
                "    data='data/r2000/data_fold0.yaml',\n",
                "    split='test',\n",
                "    batch=32,\n",
                "    imgsz=640,\n",
                "    device=0,\n",
                "    plots=True\n",
                ")\n",
                "\n",
                "p_all = float(metrics.box.mp) * 100.0\n",
                "r_all = float(metrics.box.mr) * 100.0\n",
                "map50_all = float(metrics.box.map50) * 100.0\n",
                "map50_95_all = float(metrics.box.map) * 100.0\n",
                "f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-8)\n",
                "\n",
                "# Ablation benchmark numbers for Fold 0:\n",
                "# Baseline Fold 0: P=70.08%, R=59.20%, mAP@50=67.55%, mAP50-95=49.07%, F1=63.43%\n",
                "# EMA Fold 0: P=73.25%, R=60.37%, mAP@50=69.50%, mAP50-95=50.19%, F1=66.19%\n",
                "# DySample Fold 0: P=68.11%, R=66.35%, mAP@50=68.50%, mAP50-95=49.13%, F1=67.22%\n",
                "base_p, base_r, base_map50, base_map50_95, base_f1 = 70.08, 59.20, 67.55, 49.07, 63.43\n",
                "ema_p, ema_r, ema_map50, ema_map50_95, ema_f1 = 73.25, 60.37, 69.50, 50.19, 66.19\n",
                "dys_p, dys_r, dys_map50, dys_map50_95, dys_f1 = 68.11, 66.35, 68.50, 49.13, 67.22\n",
                "\n",
                "summary_data = [\n",
                "    {\n",
                "        'Metric': 'Precision (%)',\n",
                "        'Baseline Fold 0': f'{base_p:.2f}%',\n",
                "        'EMA Fold 0': f'{ema_p:.2f}%',\n",
                "        'DySample Fold 0': f'{dys_p:.2f}%',\n",
                "        'EMA+DySample Fold 0': f'{p_all:.2f}%',\n",
                "        '\u0394 vs Baseline': f'{p_all - base_p:+.2f}%',\n",
                "        '\u0394 vs DySample': f'{p_all - dys_p:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'Recall (%)',\n",
                "        'Baseline Fold 0': f'{base_r:.2f}%',\n",
                "        'EMA Fold 0': f'{ema_r:.2f}%',\n",
                "        'DySample Fold 0': f'{dys_r:.2f}%',\n",
                "        'EMA+DySample Fold 0': f'{r_all:.2f}%',\n",
                "        '\u0394 vs Baseline': f'{r_all - base_r:+.2f}%',\n",
                "        '\u0394 vs DySample': f'{r_all - dys_r:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50 (%)',\n",
                "        'Baseline Fold 0': f'{base_map50:.2f}%',\n",
                "        'EMA Fold 0': f'{ema_map50:.2f}%',\n",
                "        'DySample Fold 0': f'{dys_map50:.2f}%',\n",
                "        'EMA+DySample Fold 0': f'{map50_all:.2f}%',\n",
                "        '\u0394 vs Baseline': f'{map50_all - base_map50:+.2f}%',\n",
                "        '\u0394 vs DySample': f'{map50_all - dys_map50:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'mAP@50-95 (%)',\n",
                "        'Baseline Fold 0': f'{base_map50_95:.2f}%',\n",
                "        'EMA Fold 0': f'{ema_map50_95:.2f}%',\n",
                "        'DySample Fold 0': f'{dys_map50_95:.2f}%',\n",
                "        'EMA+DySample Fold 0': f'{map50_95_all:.2f}%',\n",
                "        '\u0394 vs Baseline': f'{map50_95_all - base_map50_95:+.2f}%',\n",
                "        '\u0394 vs DySample': f'{map50_95_all - dys_map50_95:+.2f}%'\n",
                "    },\n",
                "    {\n",
                "        'Metric': 'F1 Score (%)',\n",
                "        'Baseline Fold 0': f'{base_f1:.2f}%',\n",
                "        'EMA Fold 0': f'{ema_f1:.2f}%',\n",
                "        'DySample Fold 0': f'{dys_f1:.2f}%',\n",
                "        'EMA+DySample Fold 0': f'{f1_all:.2f}%',\n",
                "        '\u0394 vs Baseline': f'{f1_all - base_f1:+.2f}%',\n",
                "        '\u0394 vs DySample': f'{f1_all - dys_f1:+.2f}%'\n",
                "    }\n",
                "]\n",
                "\n",
                "df_summary = pd.DataFrame(summary_data)\n",
                "print('\\n' + '=' * 90)\n",
                "print('PADDY-YOLO EMA + DYSAMPLE (CONFIG D) FOLD 0 BENCHMARK VS ABLATIONS')\n",
                "print('=' * 90)\n",
                "print(df_summary.to_string(index=False))\n",
                "df_summary.to_csv('paddy_yolo_ema_dysample_fold0_vs_ablations.csv', index=False)\n",
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
                "print('PER-SPECIES PERFORMANCE (CONFIG D FOLD 0)')\n",
                "print('=' * 90)\n",
                "print(df_species.to_string(index=False))\n",
                "df_species.to_csv('paddy_yolo_ema_dysample_fold0_species_results.csv', index=False)\n"
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

nb_path = out_dir / "paddy_yolo_ema_dysample_fold0.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook saved to: {nb_path} ({nb_path.stat().st_size} bytes)")

# 3. Kernel metadata
meta = {
    "id": "leanadrianmurillo/paddy-yolo-ema-dysample-fold0",
    "title": "paddy-yolo-ema-dysample-fold0",
    "code_file": "paddy_yolo_ema_dysample_fold0.ipynb",
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
