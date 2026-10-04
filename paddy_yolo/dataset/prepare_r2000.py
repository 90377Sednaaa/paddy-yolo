"""R2000 Dataset preparation and 5-fold cross-validation split utilities.

Complies with Section 2.1.5, Table 4, and Table 5 of the Paddy-YOLO paper.
"""

from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
import yaml


# 16 rice pest classes defined in Table 4 of Paddy-YOLO paper in exact index order
R2000_CLASSES: list[str] = [
    "rice leaf roller",
    "pink rice borer",
    "asiatic rice borer",
    "yellow rice borer",
    "rice leaf caterpillar",
    "small brown plant hopper",
    "white backed plant hopper",
    "brown plant hopper",
    "rice leaf hopper",
    "rice gall midge",
    "paddy stem maggot",
    "rice stem fly",
    "grain spreader thrips",
    "rice weevil",
    "rice shell pest",
    "melanitis leda",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def generate_dummy_dataset(
    output_dir: Union[str, Path],
    num_images: int = 20,
    img_size: Tuple[int, int] = (640, 640),
    seed: int = 42,
) -> Path:
    """Generate synthetic dummy YOLO dataset for instant testing on CPU.

    Creates output_dir/images and output_dir/labels with valid YOLO bounding boxes.

    Args:
        output_dir: Destination directory.
        num_images: Number of synthetic images to create.
        img_size: (width, height) tuple in pixels.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Path to output directory.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    labels_dir = out_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    width, height = img_size

    for idx in range(num_images):
        img_filename = f"img_{idx:04d}.jpg"
        lbl_filename = f"img_{idx:04d}.txt"

        # Create synthetic image with textured background and colored patches
        bg_color = rng.randint(30, 160, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(bg_color, mode="RGB")
        draw = ImageDraw.Draw(img)

        # Generate 1 to 3 random bounding boxes strictly within [0, 1]
        num_boxes = rng.randint(1, 4)
        label_lines: list[str] = []

        for _ in range(num_boxes):
            cls_id = int(rng.randint(0, len(R2000_CLASSES)))

            # Choose box width and height
            bw = float(rng.uniform(0.05, 0.35))
            bh = float(rng.uniform(0.05, 0.35))

            # Choose center coordinates strictly bounded
            xc = float(rng.uniform(bw / 2.0 + 0.01, 1.0 - bw / 2.0 - 0.01))
            yc = float(rng.uniform(bh / 2.0 + 0.01, 1.0 - bh / 2.0 - 0.01))

            # Draw rectangle onto image
            x1 = int((xc - bw / 2.0) * width)
            y1 = int((yc - bh / 2.0) * height)
            x2 = int((xc + bw / 2.0) * width)
            y2 = int((yc + bh / 2.0) * height)
            box_color = tuple(int(c) for c in rng.randint(100, 255, 3))
            draw.rectangle([x1, y1, x2, y2], fill=box_color, outline=(255, 255, 255), width=2)

            label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Save image and label
        img.save(images_dir / img_filename, format="JPEG", quality=90)
        (labels_dir / lbl_filename).write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return out_path


def _parse_voc_xml(xml_path: Path, img_width: int, img_height: int) -> list[str]:
    """Parse Pascal VOC XML annotation and convert to YOLO formatted strings."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        w_elem = size_elem.find("width")
        h_elem = size_elem.find("height")
        if w_elem is not None and int(w_elem.text or 0) > 0:
            img_width = int(w_elem.text)
        if h_elem is not None and int(h_elem.text or 0) > 0:
            img_height = int(h_elem.text)

    class_to_idx = {name.lower().replace("_", " "): idx for idx, name in enumerate(R2000_CLASSES)}

    yolo_lines = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        obj_name = name_elem.text.strip().lower().replace("_", " ")

        if obj_name.isdigit():
            cls_id = int(obj_name)
        elif obj_name in class_to_idx:
            cls_id = class_to_idx[obj_name]
        else:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        xc = ((xmin + xmax) / 2.0) / img_width
        yc = ((ymin + ymax) / 2.0) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def prepare_kaggle_r2000(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """Standardize raw Kaggle R2000 dataset into YOLO format (images/ and labels/).

    Supports flat directories, nested train/val/test splits, YOLO .txt labels,
    and Pascal VOC .xml annotations.

    Args:
        input_dir: Directory containing downloaded raw Kaggle files.
        output_dir: Destination standardized YOLO dataset directory.

    Returns:
        Path to prepared dataset directory.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    images_dir = out_path / "images"
    labels_dir = out_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Discover all images
    all_images = [
        p for p in in_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not all_images:
        raise FileNotFoundError(f"No image files found in {in_path}")

    for img_file in all_images:
        target_img = images_dir / img_file.name
        if not target_img.exists() or target_img.resolve() != img_file.resolve():
            shutil.copy2(img_file, target_img)

        target_lbl = labels_dir / f"{img_file.stem}.txt"

        # Search candidate label files
        candidates_txt = [
            img_file.with_suffix(".txt"),
            img_file.parent.parent / "labels" / f"{img_file.stem}.txt",
            in_path / "labels" / f"{img_file.stem}.txt",
        ]
        candidates_xml = [
            img_file.with_suffix(".xml"),
            img_file.parent.parent / "annotations" / f"{img_file.stem}.xml",
            in_path / "annotations" / f"{img_file.stem}.xml",
        ]

        found_txt = next((c for c in candidates_txt if c.is_file()), None)
        if found_txt is not None:
            if not target_lbl.exists() or target_lbl.resolve() != found_txt.resolve():
                shutil.copy2(found_txt, target_lbl)
            continue

        found_xml = next((c for c in candidates_xml if c.is_file()), None)
        if found_xml is not None:
            try:
                with Image.open(img_file) as im:
                    im_w, im_h = im.size
            except Exception:
                im_w, im_h = 640, 640
            yolo_lines = _parse_voc_xml(found_xml, im_w, im_h)
            target_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            continue

        # If no label exists, create empty label file (background image)
        if not target_lbl.exists():
            target_lbl.write_text("", encoding="utf-8")

    return out_path


def create_5fold_splits(
    dataset_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> List[str]:
    """Generate 5-fold cross validation splits matching Section 2.1.5 of the paper.

    For each fold index i in {0, 1, 2, 3, 4}:
      - Train: 3 folds
      - Val: 1 fold
      - Test: 1 fold
    Generates text files fold{i}_train.txt, fold{i}_val.txt, fold{i}_test.txt
    and dataset YAMLs data_fold{i}.yaml.

    Args:
        dataset_dir: Root dataset directory containing 'images' or image files.
        output_dir: Target directory for txt splits and YAMLs (defaults to dataset_dir).
        seed: Random seed for deterministic shuffling.

    Returns:
        List of paths to generated data_fold0.yaml ... data_fold4.yaml.
    """
    ds_path = Path(dataset_dir)
    out_path = Path(output_dir) if output_dir is not None else ds_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Find image files
    img_dir = ds_path / "images"
    search_dir = img_dir if img_dir.is_dir() else ds_path
    image_paths = sorted([
        p for p in search_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if len(image_paths) < 5:
        raise ValueError(
            f"Dataset must contain at least 5 images for 5-fold cross validation. Found {len(image_paths)}."
        )

    # Deterministic shuffling
    rng = np.random.RandomState(seed)
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)
    shuffled_paths = [image_paths[i].resolve().as_posix() for i in indices]

    # Split into 5 folds
    split_folds = [list(f) for f in np.array_split(shuffled_paths, 5)]

    yaml_paths: list[str] = []

    for fold_idx in range(5):
        # 1 fold for test, 1 fold for val, 3 folds for train
        test_images = split_folds[fold_idx]
        val_images = split_folds[(fold_idx + 1) % 5]
        train_images = (
            split_folds[(fold_idx + 2) % 5]
            + split_folds[(fold_idx + 3) % 5]
            + split_folds[(fold_idx + 4) % 5]
        )

        train_txt = out_path / f"fold{fold_idx}_train.txt"
        val_txt = out_path / f"fold{fold_idx}_val.txt"
        test_txt = out_path / f"fold{fold_idx}_test.txt"

        train_txt.write_text("\n".join(train_images) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(val_images) + "\n", encoding="utf-8")
        test_txt.write_text("\n".join(test_images) + "\n", encoding="utf-8")

        # Create dataset YAML
        yaml_content = {
            "path": out_path.resolve().as_posix(),
            "train": f"fold{fold_idx}_train.txt",
            "val": f"fold{fold_idx}_val.txt",
            "test": f"fold{fold_idx}_test.txt",
            "nc": len(R2000_CLASSES),
            "names": {i: name for i, name in enumerate(R2000_CLASSES)},
        }

        yaml_file = out_path / f"data_fold{fold_idx}.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)

        yaml_paths.append(str(yaml_file.resolve()))

    return yaml_paths


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Paddy-YOLO R2000 Dataset Preparation and 5-Fold Cross Validation Splitter"
    )
    parser.add_argument(
        "--generate-dummy",
        action="store_true",
        help="Generate synthetic dummy dataset for fast local CPU testing",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of synthetic images to create when using --generate-dummy (default: 20)",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default=None,
        help="Path to raw Kaggle R2000 dataset directory (dmrashidferrer/r2000v1-2024-images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/r2000",
        help="Output directory for processed dataset and 5-fold splits (default: dataset/r2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset generation and splitting (default: 42)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip generation of 5-fold cross validation split files",
    )
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    parser = build_argparser()
    parsed_args = parser.parse_args(args)

    out_dir = Path(parsed_args.output)

    if parsed_args.generate_dummy:
        print(f"Generating synthetic dummy dataset with {parsed_args.num_images} images at {out_dir}...")
        generate_dummy_dataset(
            output_dir=out_dir,
            num_images=parsed_args.num_images,
            seed=parsed_args.seed,
        )
        print("Dummy dataset created successfully.")
    elif parsed_args.kaggle_dir:
        print(f"Processing Kaggle R2000 dataset from {parsed_args.kaggle_dir} into {out_dir}...")
        prepare_kaggle_r2000(input_dir=parsed_args.kaggle_dir, output_dir=out_dir)
        print("Kaggle dataset standardized successfully.")
    else:
        if not (out_dir / "images").exists():
            parser.error("Specify either --generate-dummy or --kaggle-dir DIR")

    if not parsed_args.no_split:
        print(f"Generating 5-fold cross validation splits at {out_dir}...")
        yaml_paths = create_5fold_splits(dataset_dir=out_dir, seed=parsed_args.seed)
        print(f"Generated {len(yaml_paths)} fold configurations:")
        for yp in yaml_paths:
            print(f"  - {yp}")


if __name__ == "__main__":
    main()
