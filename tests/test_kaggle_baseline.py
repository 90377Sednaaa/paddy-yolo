"""Check dataset identity and leakage rejection for the Kaggle baseline."""
import importlib.util
from pathlib import Path
import shutil

from PIL import Image
import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "baseline_prepare", Path(__file__).resolve().parents[1] / "kaggle/baseline/prepare_dataset.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
prepare_dataset = _module.prepare_dataset


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    config = {"nc": 16, "names": [f"classID_{i:02d}" for i in range(16)]}
    for i, split in enumerate(("train", "val", "test")):
        config[split] = f"images/{split}"
        images, labels = root / "images" / split, root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (16, 16), (30 * i, 10, 100)).save(images / "pest.jpg")
        (labels / "pest.txt").write_text(f"{i} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return root


def test_baseline_preserves_class_ids_and_supplied_splits(source, tmp_path):
    # Duplicates within a split are allowed because annotations can differ.
    shutil.copy2(source / "images/train/pest.jpg", source / "images/train/duplicate.jpg")
    (source / "labels/train/duplicate.txt").write_text("0 0.4 0.4 0.2 0.2\n")
    prepared = prepare_dataset(source, tmp_path / "output")
    config = yaml.safe_load(prepared.read_text())
    assert config["names"] == [f"classID_{i:02d}" for i in range(16)]
    for split in ("train", "val", "test"):
        assert config[split] == f"images/{split}"
        target = prepared.parent / "labels" / split / "pest.txt"
        assert target.read_bytes() == (source / "labels" / split / "pest.txt").read_bytes()
    import json
    audit = json.loads((prepared.parent / "dataset_audit.json").read_text())
    assert audit["duplicate_image_groups"] == 1
    assert audit["cross_split_duplicates"] == 0
    assert audit["splits"]["train"]["images"] == 2
    assert audit["boxes"] == 4


def test_baseline_rejects_cross_split_duplicate(source, tmp_path):
    shutil.copy2(source / "images/train/pest.jpg", source / "images/test/pest.jpg")
    with pytest.raises(ValueError, match="Duplicate image leaks"):
        prepare_dataset(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("problem", ["missing", "invalid_class"])
def test_baseline_rejects_bad_annotations(source, tmp_path, problem):
    label = source / "labels/train/pest.txt"
    if problem == "missing":
        label.unlink()
        error = FileNotFoundError
    else:
        label.write_text("16 0.5 0.5 0.25 0.25\n")
        error = ValueError
    with pytest.raises(error):
        prepare_dataset(source, tmp_path / "output")
