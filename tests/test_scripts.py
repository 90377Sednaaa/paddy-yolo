"""Tests for top-level Paddy-YOLO CLI scripts and verification suite.

Tests train.py, val.py, predict.py, and verify_paddy_yolo.py.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml
import pytest

from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset, R2000_CLASSES


def test_verify_script_help():
    """Verify verify_paddy_yolo.py --help returns 0."""
    cmd = [sys.executable, "verify_paddy_yolo.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    assert "Paddy-YOLO" in res.stdout or "verify" in res.stdout.lower()


def test_verify_script_smoke_test():
    """Verify verify_paddy_yolo.py --smoke-test runs self-test and exits with 0."""
    cmd = [sys.executable, "verify_paddy_yolo.py", "--smoke-test"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Verification PASSED" in res.stdout or "All verification checks passed" in res.stdout


def test_train_script_help():
    """Verify train.py --help displays required CLI options."""
    cmd = [sys.executable, "train.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    for opt in ["--data", "--hyp", "--epochs", "--batch", "--fold", "--strategy"]:
        assert opt in res.stdout, f"Expected {opt} in train.py --help"


def test_val_script_help():
    """Verify val.py --help displays required CLI options."""
    cmd = [sys.executable, "val.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    for opt in ["--weights", "--data", "--imgsz", "--batch", "--split"]:
        assert opt in res.stdout, f"Expected {opt} in val.py --help"


def test_predict_script_help():
    """Verify predict.py --help displays required CLI options."""
    cmd = [sys.executable, "predict.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    for opt in ["--source", "--weights", "--imgsz", "--conf"]:
        assert opt in res.stdout, f"Expected {opt} in predict.py --help"


def test_train_script_execution(tmp_path: Path):
    """Verify train.py completes 1 epoch training run on synthetic dataset."""
    data_dir = tmp_path / "train_dummy"
    generate_dummy_dataset(data_dir, num_images=4, img_size=(64, 64))

    data_yaml = data_dir / "data.yaml"
    yaml_content = {
        "path": str(data_dir.resolve()).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "nc": len(R2000_CLASSES),
        "names": {i: name for i, name in enumerate(R2000_CLASSES)},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)

    project_dir = tmp_path / "runs"
    cmd = [
        sys.executable,
        "train.py",
        "--data", str(data_yaml),
        "--model", "configs/paddy-yolo.yaml",
        "--epochs", "1",
        "--batch", "2",
        "--imgsz", "64",
        "--device", "cpu",
        "--workers", "0",
        "--project", str(project_dir),
        "--name", "test_train",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_val_script_execution(tmp_path: Path):
    """Verify val.py evaluates Paddy-YOLO on synthetic dataset and outputs metrics."""
    data_dir = tmp_path / "val_dummy"
    generate_dummy_dataset(data_dir, num_images=4, img_size=(64, 64))

    data_yaml = data_dir / "data.yaml"
    yaml_content = {
        "path": str(data_dir.resolve()).replace("\\", "/"),
        "train": "images",
        "val": "images",
        "nc": len(R2000_CLASSES),
        "names": {i: name for i, name in enumerate(R2000_CLASSES)},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)

    project_dir = tmp_path / "val_runs"
    cmd = [
        sys.executable,
        "val.py",
        "--model", "configs/paddy-yolo.yaml",
        "--data", str(data_yaml),
        "--imgsz", "64",
        "--batch", "2",
        "--device", "cpu",
        "--project", str(project_dir),
        "--name", "test_val",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "mAP" in res.stdout or "mAP50" in res.stdout or "Validation completed" in res.stdout


def test_predict_script_execution(tmp_path: Path):
    """Verify predict.py runs inference on an image and writes output."""
    data_dir = tmp_path / "pred_dummy"
    generate_dummy_dataset(data_dir, num_images=2, img_size=(64, 64))
    img_file = next((data_dir / "images").glob("*.jpg"))

    output_dir = tmp_path / "pred_runs"
    cmd = [
        sys.executable,
        "predict.py",
        "--model", "configs/paddy-yolo.yaml",
        "--source", str(img_file),
        "--imgsz", "64",
        "--device", "cpu",
        "--project", str(output_dir),
        "--name", "test_pred",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "Prediction completed" in res.stdout or "Results saved" in res.stdout
