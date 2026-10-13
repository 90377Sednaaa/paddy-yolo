"""Build a separate private Kaggle notebook for five fresh EMA-head folds."""
import argparse
import json
from pathlib import Path

from build_dysample_crossval_kernel import build as build_fivefold


def build(owner="dmrashidpferrer", output="kaggle_ema_heads_fivefold_kernel"):
    output = build_fivefold(owner, output)
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_path = output / metadata["code_file"]
    notebook = json.loads(old_path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][-1]["source"])
    source = source.replace("# Five independent plain DySample folds, 300 epochs each.",
                            "# Five independent EMA detection-branch folds, 300 epochs each.")
    source = source.replace("VARIANT = 'proposed'", "VARIANT = 'ema_heads'")
    source = source.replace("'/kaggle/working/dysample_fivefold'", "'/kaggle/working/ema_heads_fivefold'")
    source = source.replace("'--start-fold', str(START_FOLD)]",
                            "'--start-fold', str(START_FOLD), '--variant', 'ema_heads']")
    source = source.replace("Starting plain DySample five-fold training", "Starting EMA-head five-fold training")
    notebook["cells"][-1]["source"] = source.splitlines(keepends=True)
    notebook["cells"][0]["source"] = [
        "# Paddy-YOLO: EMA before P3/P4/P5, five-fold R2000 experiment\n",
        "Nearest-neighbor neck; three residual EMA attention branches, eight channel groups each; "
        "unchanged RGG backbone and DIMIoU Strategy H. Every fold starts from compatible yolov8n.pt "
        "weights and trains for 300 epochs, global batch 32, 640px, AdamW, seed 42, "
        "checkpoint period 50. Two T4 GPUs use DDP when available. "
        "Best checkpoints receive held-out test evaluation and exact per-class reporting.\n",
    ]
    metadata.update(id=f"{owner}/paddy-yolo-ema-heads-r2000-fivefold",
                    title="Paddy YOLO EMA Heads R2000 Fivefold", code_file="paddy_yolo_ema_heads_r2000.ipynb")
    (output / metadata["code_file"]).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    old_path.unlink()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="dmrashidpferrer")
    parser.add_argument("--output", default="kaggle_ema_heads_fivefold_kernel")
    args = parser.parse_args()
    print(build(args.owner, args.output))
