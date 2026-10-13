"""Explicit checkpoint saving and resume selection for Kaggle sessions."""

from pathlib import Path

import pytest

from train import parse_args


def test_save_period_keeps_existing_defaults():
    defaults = parse_args([])
    assert defaults.save_period is None
    assert defaults.resume_from is None
    args = parse_args(["--model", "configs/Paddy-YOLO-DySample.yaml", "--weights", "yolov8n.pt",
                       "--batch", "32", "--device", "0,1", "--save-period", "50"])
    assert args.save_period == 50
    assert args.device == "0,1"
    assert args.batch == 32


def test_resume_from_existing_checkpoint(tmp_path):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = parse_args(["--resume-from", str(checkpoint)])
    assert args.resume_from == str(checkpoint)


def test_conflicting_initialization_and_resume_is_rejected(tmp_path):
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    from train import resolve_resume_checkpoint
    with pytest.raises(ValueError, match="initial weights"):
        resolve_resume_checkpoint("yolov8n.pt", str(checkpoint))
    assert resolve_resume_checkpoint(None, str(checkpoint)) == str(checkpoint.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(None, str(tmp_path / "missing.pt"))
