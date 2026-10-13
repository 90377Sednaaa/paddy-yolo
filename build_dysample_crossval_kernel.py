"""Build a separate private Kaggle notebook for all five plain DySample folds."""
import argparse
import json
from pathlib import Path

from build_dysample_full_kernel import build as build_full
from build_dysample_kaggle_kernel import _cell


def build(owner="dmrashidferrer", output="kaggle_dysample_fivefold_kernel"):
    output = build_full(owner, 0, output, variant="proposed")
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    notebook_path = output / metadata["code_file"]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    original = "".join(notebook["cells"][-1]["source"])
    discovery = original.split("command = [", 1)[0]
    discovery = discovery.replace("# Full 300-epoch R2000 fold 0 training; no synthetic preflight in this notebook.",
                                  "# Five independent plain DySample folds, 300 epochs each.")
    discovery = discovery.replace("FOLD = 0", "START_FOLD = 0")
    training = discovery + '''command = [sys.executable, 'run_dysample_crossval.py',
           '--raw-data', str(raw_data), '--output', '/kaggle/working/dysample_fivefold',
           '--start-fold', str(START_FOLD)]
if RESUME_FROM:
    command += ['--resume-from', RESUME_FROM]
print('Starting plain DySample five-fold training and held-out test evaluation', flush=True)
subprocess.run(command, check=True)
'''
    notebook["cells"][-1] = _cell(training)
    notebook["cells"][0]["source"] = [
        "# Plain Paddy-YOLO-DySample: five-fold R2000 experiment\n",
        "Every fold initializes from yolov8n.pt and trains for 300 epochs, global batch 32, "
        "640px, DIMIoU and AdamW. Two T4 GPUs use DDP when available. "
        "Each best checkpoint is evaluated on its held-out test split; exact per-class "
        "results and five-fold mean/sample standard deviation are saved.\n",
    ]
    metadata["id"] = f"{owner}/paddy-yolo-dysample-r2000-fivefold"
    metadata["title"] = "Paddy YOLO DySample R2000 Fivefold"
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="dmrashidferrer")
    parser.add_argument("--output", default="kaggle_dysample_fivefold_kernel")
    args = parser.parse_args()
    print(build(args.owner, args.output))
