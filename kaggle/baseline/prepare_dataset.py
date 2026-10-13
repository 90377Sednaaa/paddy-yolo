"""Audit R2000-PestNet and stage its existing splits in a writable directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image
import yaml

DATASET = "dmrashidpferrer/r2000-pestnet"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dataset(input_dir: Path) -> Path:
    """Accept a dataset root or discover exactly one attached R2000 dataset."""
    input_dir = input_dir.resolve()
    if (input_dir / "data.yaml").is_file():
        return input_dir
    matches = [p.parent for p in input_dir.rglob("data.yaml")
               if "r2000-pestnet" in p.as_posix().lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one attached {DATASET} dataset, found {len(matches)}. "
            "Add it with Kaggle's Add Input button or pass --input-dir explicitly."
        )
    return matches[0]


def prepare_dataset(input_dir: Path, output_dir: Path) -> Path:
    """Preserve split assignments and class IDs; reject corrupt data or leakage."""
    source = find_dataset(input_dir)
    destination = output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output directories must be separate, non-nested directories.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Use a new, empty output directory: {destination}")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if len(names) != 16 or config.get("nc", len(names)) != 16:
        raise ValueError("R2000-PestNet must contain 16 classes.")

    report = {"dataset": DATASET, "names": names, "splits": {}, "boxes": 0,
              "duplicate_image_groups": 0, "cross_split_duplicates": 0}
    seen_hashes = {}
    staged = []
    for split in ("train", "val", "test"):
        relative = Path(config[split])
        if relative.as_posix() != f"images/{split}":
            raise ValueError(f"Expected images/{split}, got {relative}")
        images = sorted(p for p in (source / relative).rglob("*")
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"No images in {split} split")
        boxes = 0
        class_counts = [0] * 16
        stems = set()
        expected_labels = set()
        for image in images:
            subpath = image.relative_to(source / relative)
            stem = subpath.with_suffix("").as_posix()
            if stem in stems:
                raise ValueError(f"Multiple images use the same label: {image}")
            stems.add(stem)
            label = source / "labels" / split / subpath.with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing annotation: {label}")
            expected_labels.add(label.resolve())
            with Image.open(image) as im:
                im.verify()
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen_hashes and seen_hashes[digest][0] != split:
                raise ValueError(f"Duplicate image leaks between {seen_hashes[digest][0]} and {split}: {image}")
            previous = seen_hashes.get(digest, (split, 0))
            seen_hashes[digest] = (split, previous[1] + 1)
            for line_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not row.strip():
                    continue
                tokens = row.split()
                if len(tokens) != 5:
                    raise ValueError(f"Expected 5 YOLO values in {label}:{line_number}")
                class_id = int(tokens[0])
                x, y, w, h = map(float, tokens[1:])
                if not 0 <= class_id < 16 or not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError(f"Invalid class or coordinates in {label}:{line_number}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Invalid normalized box in {label}:{line_number}")
                # Permit rounding at six decimal places along image boundaries.
                if min(x - w / 2, y - h / 2) < -1e-5 or max(x + w / 2, y + h / 2) > 1 + 1e-5:
                    raise ValueError(f"Box exceeds image bounds in {label}:{line_number}")
                boxes += 1
                class_counts[class_id] += 1
            staged.append((image, label, split, subpath))
        actual_labels = {p.resolve() for p in (source / "labels" / split).rglob("*.txt")}
        if actual_labels != expected_labels:
            raise ValueError(f"Unmatched annotation files in {split} split")
        if boxes == 0:
            raise ValueError(f"No annotated boxes in {split} split")
        report["splits"][split] = {"images": len(images), "boxes": boxes, "class_boxes": class_counts}
        report["boxes"] += boxes
    report["duplicate_image_groups"] = sum(count > 1 for _, count in seen_hashes.values())

    # Copy small annotations and link images to keep caches writable on Kaggle.
    for image, label, split, subpath in staged:
        target_image = destination / "images" / split / subpath
        target_label = destination / "labels" / split / subpath.with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(image.resolve(), target_image)
        except OSError:
            shutil.copy2(image, target_image)
        shutil.copy2(label, target_label)
    prepared = {"path": destination.as_posix(), "train": "images/train",
                "val": "images/val", "test": "images/test", "nc": 16, "names": names}
    data_yaml = destination / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(prepared, sort_keys=False), encoding="utf-8")
    (destination / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return data_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/baseline_data"))
    args = parser.parse_args()
    print("Prepared config:", prepare_dataset(args.input_dir, args.output_dir))
