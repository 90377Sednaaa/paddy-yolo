"""Build a private fivefold baseline notebook; never includes API credentials."""
import argparse
import json
import runpy
from pathlib import Path


def build(owner, slug="paddy-yolo-baseline-fivefold-r2000-20261011", experiment="baseline"):
    runpy.run_path("build_kaggle_kernel.py")
    out = Path("kaggle_kernel")
    old = json.loads((out / "paddy_yolo_baseline_training.ipynb").read_text(encoding="utf-8"))
    extract = "".join(old["cells"][1]["source"])
    setup = '''import os, subprocess, sys
from pathlib import Path
os.chdir('/kaggle/working')
os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/.ultralytics'
subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.3.253', '--quiet'], check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU required')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
configs = [p for p in Path('/kaggle/input').rglob('data.yaml') if 'r2000' in str(p).lower()]
if len(configs) != 1:
    raise RuntimeError(f'Expected one mounted R2000 data.yaml; found {configs}')
subprocess.run([sys.executable, 'run_baseline_crossval.py', '--raw-data', str(configs[0].parent),
                '--output', '/kaggle/working/baseline_results'], check=True)
'''
    old["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# YOLOv8-N baseline: fivefold R2000 replication\n",
            "300 epochs per fold, batch 32, 640px, AdamW, CIoU, random initialization.\n",
            "Three training folds, one validation fold, one held-out test fold.\n",
            "Validation selects best.pt; report the mean and sample standard deviation over five test folds.\n",
            "Corrective rerun: FP32 validation, AdamW bias warmup 0.0, and strict finite-loss checks.\n",
            "Replication assumptions are included in docs/replication-audit.md.\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": (extract + "\n" + setup).splitlines(keepends=True)},
    ]
    if experiment == "rgg-impdiou":
        source = "".join(old["cells"][1]["source"])
        source = source.replace("'--output', '/kaggle/working/baseline_results'", "'--output', '/kaggle/working/rgg_impdiou_results', '--experiment', 'rgg-impdiou'")
        old["cells"][1]["source"] = source.splitlines(keepends=True)
        old["cells"][0]["source"] = [
            "# RGG + IMPDIoU: paper R2000 fivefold experiment\n",
            "Four RGG backbone blocks; fixed IMPDIoU ratio 1.3; BCE classification and DFL retained.\n",
            "300 epochs per fold, batch 32, 640px, AdamW, random initialization, FP32 validation.\n",
            "Comparison: Table 13 + RGG + IMPD, mAP50 69.7%, mAP50:95 50.8%.\n",
            "Runtime preflight checks both the RGG architecture and the installed fixed-ratio loss.\n",
        ]
    elif experiment == "rgg-dimiou":
        source = "".join(old["cells"][1]["source"])
        source = source.replace("'--output', '/kaggle/working/baseline_results'", "'--output', '/kaggle/working/rgg_dimiou_results', '--experiment', 'rgg-dimiou'")
        old["cells"][1]["source"] = source.splitlines(keepends=True)
        old["cells"][0]["source"] = [
            "# Paddy-YOLO: RGG + DIMIoU Strategy H, fivefold R2000 replication\n",
            "Four RGG backbone blocks; dynamic DIMIoU ratio, Strategy H; BCE classification and DFL retained.\n",
            "300 epochs per fold, batch 32, 640px, AdamW, random initialization, FP32 validation.\n",
            "Comparison: Tables 8 and 13 final Paddy-YOLO, mAP50 70.2%, mAP50:95 51.5%.\n",
            "Runtime preflight checks four RGG blocks, Strategy H, and absence of a fixed ratio.\n",
        ]
    filename = "baseline_fivefold.ipynb"
    (out / filename).write_text(json.dumps(old, indent=2), encoding="utf-8")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    meta.update(id=f"{owner}/{slug}", title=slug.replace("-", " "),
                code_file=filename, is_private=True, enable_gpu=True, enable_internet=True)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Prepared private GPU notebook:", meta["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", default="paddy-yolo-baseline-fivefold-r2000-20261011")
    parser.add_argument("--experiment", choices=["baseline", "rgg-impdiou", "rgg-dimiou"], default="baseline")
    args = parser.parse_args()
    build(args.owner, args.slug, args.experiment)
