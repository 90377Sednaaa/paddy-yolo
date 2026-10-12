import os
import subprocess
import sys
from pathlib import Path
import yaml
from PIL import Image

import pytest
from ultralytics.data.utils import check_det_dataset

from paddy_yolo.dataset import (
    R2000_CLASSES,
    create_5fold_splits,
    generate_dummy_dataset,
    prepare_kaggle_r2000,
)


EXPECTED_CLASSES = [
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


def test_source_class_order_is_remapped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (32, 32)).save(raw / "one.jpg")
    (raw / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": list(reversed(EXPECTED_CLASSES))}))
    out = prepare_kaggle_r2000(raw, tmp_path / "out")
    assert (out / "labels/one.txt").read_text().startswith("15 ")


def test_users_nested_kaggle_layout_and_generic_class_names(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images/train").mkdir(parents=True)
    (raw / "labels/train").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(raw / "images/train/08_00001_.jpg")
    (raw / "labels/train/08_00001_.txt").write_text("08 0.5 0.5 0.2 0.2\n")
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": {i: f"classID_{i:02}" for i in range(16)}}))
    out = prepare_kaggle_r2000(raw, tmp_path / "out")
    assert (out / "labels/08_00001_.txt").read_text().startswith("8 ")


def test_duplicate_images_stay_in_the_same_crossvalidation_fold(tmp_path):
    import shutil
    generate_dummy_dataset(tmp_path, num_images=9, img_size=(32, 32))
    shutil.copyfile(tmp_path / "images/img_0000.jpg", tmp_path / "images/duplicate.jpg")
    shutil.copyfile(tmp_path / "labels/img_0000.txt", tmp_path / "labels/duplicate.txt")
    create_5fold_splits(tmp_path)
    original = (tmp_path / "images/img_0000.jpg").as_posix()
    duplicate = (tmp_path / "images/duplicate.jpg").as_posix()
    for i in range(5):
        for split in ("train", "val", "test"):
            paths = (tmp_path / f"fold{i}_{split}.txt").read_text().splitlines()
            assert (original in paths) == (duplicate in paths)


def test_flattening_duplicate_stems_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    for folder in ("train", "val"):
        directory = raw / folder
        directory.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(directory / "same.jpg")
    with pytest.raises(ValueError, match="Duplicate image stem"):
        prepare_kaggle_r2000(raw, tmp_path / "out")


def test_missing_annotation_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (32, 32)).save(raw / "one.jpg")
    with pytest.raises(FileNotFoundError, match="No annotation"):
        prepare_kaggle_r2000(raw, tmp_path / "out")


def test_revised_github_extra_classes_are_rejected_without_yaml(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (32, 32)).save(raw / "one.jpg")
    (raw / "one.txt").write_text("16 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="Invalid class ID"):
        prepare_kaggle_r2000(raw, tmp_path / "out")


def test_splits_preserve_symlink_label_paths(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    generate_dummy_dataset(raw, num_images=5, img_size=(32, 32))
    out = prepare_kaggle_r2000(raw, tmp_path / "prepared")
    create_5fold_splits(out)
    paths = (out / "fold0_test.txt").read_text().splitlines()
    assert all(Path(p).parent == out / "images" for p in paths)
    assert all((out / "labels" / (Path(p).stem + ".txt")).exists() for p in paths)


def test_r2000_classes_table4_compliance():
    """Verify class list strictly adheres to Table 4 of Paddy-YOLO paper."""
    assert len(R2000_CLASSES) == 16
    assert R2000_CLASSES == EXPECTED_CLASSES


def test_generate_dummy_dataset(tmp_path):
    """Verify synthetic dummy dataset creates valid YOLO images and bounded labels."""
    data_dir = tmp_path / "dummy_data"
    num_images = 20
    generate_dummy_dataset(str(data_dir), num_images=num_images, img_size=(640, 640), seed=42)

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    assert images_dir.is_dir()
    assert labels_dir.is_dir()

    images = list(images_dir.glob("*.jpg"))
    labels = list(labels_dir.glob("*.txt"))

    assert len(images) == num_images
    assert len(labels) == num_images

    # Verify each image can be opened and has correct dimensions
    for img_path in images:
        with Image.open(img_path) as img:
            assert img.size == (640, 640)
            assert img.mode == "RGB"

    # Verify label format: class x_c y_c w h in [0, 1]
    for lbl_path in labels:
        content = lbl_path.read_text().strip()
        assert len(content) > 0, f"Label file {lbl_path} should not be empty"
        lines = content.splitlines()
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 values per bbox, got {line}"
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            assert 0 <= cls_id < 16
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0
            assert 0.0 <= xc - w / 2.0
            assert xc + w / 2.0 <= 1.0
            assert 0.0 <= yc - h / 2.0
            assert yc + h / 2.0 <= 1.0


def test_create_5fold_splits(tmp_path):
    """Verify 5-fold cross validation split generation per Section 2.1.5."""
    data_dir = tmp_path / "dummy_data"
    generate_dummy_dataset(str(data_dir), num_images=20, seed=42)

    yaml_paths = create_5fold_splits(str(data_dir), seed=42)
    assert len(yaml_paths) == 5

    all_images = {p.resolve().as_posix() for p in (data_dir / "images").glob("*.jpg")}
    assert len(all_images) == 20

    for i, yaml_file in enumerate(yaml_paths):
        yaml_path = Path(yaml_file)
        assert yaml_path.exists()
        assert yaml_path.name == f"data_fold{i}.yaml"

        train_txt = yaml_path.parent / f"fold{i}_train.txt"
        val_txt = yaml_path.parent / f"fold{i}_val.txt"
        test_txt = yaml_path.parent / f"fold{i}_test.txt"

        assert train_txt.exists()
        assert val_txt.exists()
        assert test_txt.exists()

        train_imgs = set(filter(None, train_txt.read_text().splitlines()))
        val_imgs = set(filter(None, val_txt.read_text().splitlines()))
        test_imgs = set(filter(None, test_txt.read_text().splitlines()))

        # Non-overlapping verification
        assert len(val_imgs & test_imgs) == 0, f"Fold {i}: Val and Test overlap!"
        assert len(train_imgs & val_imgs) == 0, f"Fold {i}: Train and Val overlap!"
        assert len(train_imgs & test_imgs) == 0, f"Fold {i}: Train and Test overlap!"

        # 3 folds train (12), 1 fold val (4), 1 fold test (4) for 20 images
        assert len(train_imgs) == 12, f"Fold {i}: Expected 12 train images, got {len(train_imgs)}"
        assert len(val_imgs) == 4, f"Fold {i}: Expected 4 val images, got {len(val_imgs)}"
        assert len(test_imgs) == 4, f"Fold {i}: Expected 4 test images, got {len(test_imgs)}"

        # Union covers all images
        assert train_imgs | val_imgs | test_imgs == all_images

        # Verify YAML content and compliance
        with open(yaml_path) as f:
            data_cfg = yaml.safe_load(f)

        assert data_cfg["nc"] == 16
        assert data_cfg["train"] == f"fold{i}_train.txt"
        assert data_cfg["val"] == f"fold{i}_val.txt"
        assert data_cfg["test"] == f"fold{i}_test.txt"
        assert [data_cfg["names"][idx] for idx in range(16)] == EXPECTED_CLASSES

        # Ultralytics dataset integrity check
        validated = check_det_dataset(str(yaml_path))
        assert validated["nc"] == 16


def test_create_5fold_splits_custom_output(tmp_path):
    """Verify create_5fold_splits respects custom output directory."""
    data_dir = tmp_path / "data"
    splits_dir = tmp_path / "splits"
    generate_dummy_dataset(str(data_dir), num_images=15, seed=42)

    yaml_paths = create_5fold_splits(str(data_dir), output_dir=str(splits_dir), seed=42)
    assert len(yaml_paths) == 5
    for yaml_path in yaml_paths:
        assert Path(yaml_path).parent == splits_dir
        assert Path(yaml_path).exists()


def test_prepare_kaggle_r2000(tmp_path):
    """Verify Kaggle R2000 structure is standardized into images/ and labels/."""
    raw_kaggle = tmp_path / "raw_kaggle"
    raw_kaggle.mkdir()

    # Simulate raw Kaggle files: image with annotation file
    img = Image.new("RGB", (640, 640), color="green")
    img.save(raw_kaggle / "sample1.jpg")
    (raw_kaggle / "sample1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    img.save(raw_kaggle / "sample2.png")
    (raw_kaggle / "sample2.txt").write_text("1 0.4 0.4 0.3 0.3\n")

    out_dir = tmp_path / "prepared"
    prepare_kaggle_r2000(str(raw_kaggle), str(out_dir))

    assert (out_dir / "images" / "sample1.jpg").exists()
    assert (out_dir / "images" / "sample2.png").exists()
    assert (out_dir / "labels" / "sample1.txt").read_text().strip() == "0 0.5 0.5 0.2 0.2"
    assert (out_dir / "labels" / "sample2.txt").read_text().strip() == "1 0.4 0.4 0.3 0.3"


def test_prepare_r2000_cli(tmp_path):
    """Verify CLI interface with --generate-dummy and 5-fold split generation."""
    out_dir = tmp_path / "cli_dummy"
    cmd = [
        sys.executable,
        "-m",
        "paddy_yolo.dataset.prepare_r2000",
        "--generate-dummy",
        "--output",
        str(out_dir),
        "--seed",
        "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

    assert (out_dir / "images").exists()
    assert (out_dir / "labels").exists()
    for i in range(5):
        assert (out_dir / f"data_fold{i}.yaml").exists()
        assert (out_dir / f"fold{i}_train.txt").exists()
        assert (out_dir / f"fold{i}_val.txt").exists()
        assert (out_dir / f"fold{i}_test.txt").exists()


def test_cross_validation_full_coverage(tmp_path):
    """Verify every sample appears in test exactly once, val once, train 3 times."""
    data_dir = tmp_path / "coverage_data"
    generate_dummy_dataset(str(data_dir), num_images=20, seed=123)
    yaml_paths = create_5fold_splits(str(data_dir), seed=123)

    from collections import Counter
    test_counter = Counter()
    val_counter = Counter()
    train_counter = Counter()

    for yaml_file in yaml_paths:
        fold_dir = Path(yaml_file).parent
        stem = Path(yaml_file).stem.replace("data_", "")
        test_txt = fold_dir / f"{stem}_test.txt"
        val_txt = fold_dir / f"{stem}_val.txt"
        train_txt = fold_dir / f"{stem}_train.txt"

        for line in test_txt.read_text().splitlines():
            if line:
                test_counter[line] += 1
        for line in val_txt.read_text().splitlines():
            if line:
                val_counter[line] += 1
        for line in train_txt.read_text().splitlines():
            if line:
                train_counter[line] += 1

    all_images = [p.resolve().as_posix() for p in (data_dir / "images").glob("*.jpg")]
    assert len(all_images) == 20

    for img in all_images:
        assert test_counter[img] == 1, f"{img} appeared in test {test_counter[img]} times"
        assert val_counter[img] == 1, f"{img} appeared in val {val_counter[img]} times"
        assert train_counter[img] == 3, f"{img} appeared in train {train_counter[img]} times"


def test_create_5fold_splits_minimum_samples(tmp_path):
    """Verify ValueError is raised when dataset has fewer than 5 images."""
    data_dir = tmp_path / "too_few"
    generate_dummy_dataset(str(data_dir), num_images=4, seed=42)

    with pytest.raises(ValueError, match="at least 5 images"):
        create_5fold_splits(str(data_dir))


def test_prepare_kaggle_voc_xml(tmp_path):
    """Verify Pascal VOC XML annotations are accurately parsed into normalized YOLO."""
    raw_kaggle = tmp_path / "raw_voc"
    raw_kaggle.mkdir()

    # Create dummy image
    img = Image.new("RGB", (640, 480), color="white")
    img.save(raw_kaggle / "pest_voc.jpg")

    # Create corresponding VOC XML
    xml_content = """<annotation>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <object>
        <name>pink_rice_borer</name>
        <bndbox>
            <xmin>64</xmin>
            <ymin>48</ymin>
            <xmax>192</xmax>
            <ymax>144</ymax>
        </bndbox>
    </object>
</annotation>"""
    (raw_kaggle / "pest_voc.xml").write_text(xml_content)

    out_dir = tmp_path / "prepared_voc"
    prepare_kaggle_r2000(str(raw_kaggle), str(out_dir))

    lbl_file = out_dir / "labels" / "pest_voc.txt"
    assert lbl_file.exists()
    content = lbl_file.read_text().strip()
    parts = content.split()
    assert len(parts) == 5
    cls_id, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    assert cls_id == 1  # pink rice borer is index 1 in Table 4
    # Expected: xmin=64, xmax=192 -> xc = (64+192)/(2*640) = 128/640 = 0.2
    # w = (192-64)/640 = 128/640 = 0.2
    # ymin=48, ymax=144 -> yc = (48+144)/(2*480) = 96/480 = 0.2
    # h = (144-48)/480 = 96/480 = 0.2
    assert pytest.approx(xc, rel=1e-3) == 0.2
    assert pytest.approx(yc, rel=1e-3) == 0.2
    assert pytest.approx(w, rel=1e-3) == 0.2
    assert pytest.approx(h, rel=1e-3) == 0.2

