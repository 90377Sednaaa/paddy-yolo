"""The Kaggle package must be reproducible and safe to run as a preflight."""

import io
import json
from types import SimpleNamespace
import zipfile

import pytest

from build_dysample_kaggle_kernel import build, source_archive
from build_dysample_full_kernel import build as build_full
from run_dysample_experiment import training_command, validate_resume_checkpoint


def test_source_archive_contains_model_and_license_only():
    with zipfile.ZipFile(io.BytesIO(source_archive())) as archive:
        names = set(archive.namelist())
        assert "configs/Paddy-YOLO-DySample.yaml" in names
        assert "configs/paddy-yolo.yaml" in names
        assert "paddy_yolo/modules/dysample.py" in names
        assert "paddy_yolo/modules/DySample-LICENSE" in names
        assert "verify_dysample.py" in names
        assert "run_dysample_experiment.py" in names
        assert all("kaggle.json" not in name.lower() for name in names)
        assert all(not name.endswith(".pt") for name in names)


def test_generated_kernel_is_private_preflight_and_training_is_gated(tmp_path):
    out = build("dmrashidferrer", tmp_path)
    metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((out / metadata["code_file"]).read_text(encoding="utf-8"))
    assert metadata["id"].startswith("dmrashidferrer/")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        compile(cell, "notebook-cell", "exec")
    assert any("verify_dysample.py" in cell and "pytest" in cell for cell in cells)
    assert any("--mini-ddp" in cell for cell in cells)
    assert "PYTHONPATH" in cells[0]
    assert "'/kaggle/working'" in cells[0]
    training = next(cell for cell in cells if "RUN_FULL_TRAINING" in cell)
    assert "RUN_FULL_TRAINING = False" in training
    assert "if RUN_FULL_TRAINING:" in training
    assert "--variant" in training
    assert "--resume-from" in training


def test_training_commands_keep_variants_separate_and_resume_without_reinitialization(tmp_path):
    baseline = training_command("baseline", 0, "fold0.yaml", tmp_path, 2)
    proposed = training_command("proposed", 0, "fold0.yaml", tmp_path, 2,
                                "/kaggle/input/old-run/last.pt")
    assert baseline[baseline.index("--model") + 1] == "configs/paddy-yolo.yaml"
    assert proposed[proposed.index("--model") + 1] == "configs/Paddy-YOLO-DySample.yaml"
    assert baseline[baseline.index("--device") + 1] == "0,1"
    assert baseline[baseline.index("--batch") + 1] == "32"
    assert baseline[baseline.index("--save-period") + 1] == "50"
    assert baseline[baseline.index("--weights") + 1] == "yolov8n.pt"
    assert "--weights" not in proposed
    assert proposed[proposed.index("--resume-from") + 1].endswith("last.pt")
    assert baseline[baseline.index("--project") + 1] != proposed[proposed.index("--project") + 1]


def test_resume_rejects_wrong_variant_fold_and_stripped_state(tmp_path, monkeypatch):
    checkpoint = tmp_path / "epoch50.pt"
    checkpoint.touch()
    saved = {
        "train_args": {"data": "/kaggle/working/data/r2000/data_fold0.yaml",
                       "project": "/kaggle/working/dysample_training/proposed",
                       "name": "fold0", "epochs": 300},
        "ema": SimpleNamespace(yaml={"head": [
            [-1, 1, "DySample", [2, "lp", 4, False]], [0, 1, "Concat", [1]],
            [-1, 1, "C2f", [128]], [-1, 1, "DySample", [2, "lp", 4, False]],
        ]}),
        "optimizer": {"state": {}}, "scaler": {}, "epoch": 50,
    }
    monkeypatch.setattr("run_dysample_experiment.torch.load", lambda *a, **kw: saved)
    validate_resume_checkpoint(checkpoint, "proposed", 0)
    with pytest.raises(ValueError, match="architecture"):
        validate_resume_checkpoint(checkpoint, "baseline", 0)
    with pytest.raises(ValueError, match="fold"):
        validate_resume_checkpoint(checkpoint, "proposed", 1)
    saved["optimizer"] = None
    with pytest.raises(ValueError, match="optimizer"):
        validate_resume_checkpoint(checkpoint, "proposed", 0)


def test_full_kernel_runs_proposed_r2000_without_preflight(tmp_path):
    out = build_full("dmrashidferrer", 0, tmp_path)
    metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((out / metadata["code_file"]).read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        compile(cell, "full-kernel-cell", "exec")
    assert metadata["id"] == "dmrashidferrer/paddy-yolo-dysample-r2000-fold0"
    assert metadata["is_private"] is True
    assert metadata["dataset_sources"] == ["dmrashidpferrer/r2000-pestnet"]
    assert "PYTHONPATH" in cells[0]
    assert "VARIANT = 'proposed'" in cells[1]
    assert "FOLD = 0" in cells[1]
    assert "run_dysample_experiment.py" in cells[1]
    assert all("verify_dysample.py" not in cell and "pytest" not in cell for cell in cells)


def test_plus_training_command_uses_separate_architecture_and_outputs(tmp_path):
    command = training_command("proposed_plus", 0, "fold0.yaml", tmp_path, 2)
    options = dict(zip(command[2::2], command[3::2]))
    assert options["--model"] == "configs/Paddy-YOLO-DySample-Plus.yaml"
    assert options["--project"] == str(tmp_path / "proposed_plus")
    assert options["--weights"] == "yolov8n.pt"
    assert options["--hyp"] == "configs/hyp_paper.yaml"
    assert options["--epochs"] == "300"
    assert options["--batch"] == "32"
    assert options["--device"] == "0,1"
    assert options["--save-period"] == "50"


def test_plus_kernel_has_distinct_identity_and_packages_scope_architecture(tmp_path):
    import ast
    import base64
    import yaml

    out = build_full("dmrashidferrer", 0, tmp_path, variant="proposed_plus")
    metadata = json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((out / metadata["code_file"]).read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert metadata["id"] == "dmrashidferrer/paddy-yolo-dysample-plus-r2000-fold0"
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    for cell in cells:
        compile(cell, "plus-notebook-cell", "exec")
    assert "VARIANT = 'proposed_plus'" in cells[1]
    tree = ast.parse(cells[0])
    encoded = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "B64_ARCHIVE" for target in node.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        config = yaml.safe_load(archive.read("configs/Paddy-YOLO-DySample-Plus.yaml"))
        assert [config["head"][i][3] for i in (0, 3)] == [[2, "lp", 4, True]] * 2


def test_plus_resume_rejects_plain_dysample_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "epoch50.pt"
    checkpoint.touch()
    saved = {
        "train_args": {"data": "data_fold0.yaml", "project": "proposed_plus", "name": "fold0"},
        "ema": SimpleNamespace(yaml={"head": [
            [-1, 1, "DySample", [2, "lp", 4, False]], [0, 1, "Concat", [1]],
            [-1, 1, "C2f", [128]], [-1, 1, "DySample", [2, "lp", 4, False]],
        ]}),
        "optimizer": {"state": {}}, "scaler": {}, "epoch": 50,
    }
    monkeypatch.setattr("run_dysample_experiment.torch.load", lambda *a, **kw: saved)
    with pytest.raises(ValueError, match="architecture"):
        validate_resume_checkpoint(checkpoint, "proposed_plus", 0)
    for i in (0, 3):
        saved["ema"].yaml["head"][i][3][-1] = True
    validate_resume_checkpoint(checkpoint, "proposed_plus", 0)


@pytest.mark.parametrize("entrypoint", ["python", "cli"])
def test_default_plus_build_preserves_plain_notebook(tmp_path, monkeypatch, entrypoint):
    from pathlib import Path
    import subprocess
    import sys

    builder = Path(__file__).resolve().parents[1] / "build_dysample_full_kernel.py"
    monkeypatch.chdir(tmp_path)
    plain = build_full("dmrashidferrer")
    original_metadata = (plain / "kernel-metadata.json").read_bytes()
    if entrypoint == "python":
        build_full("dmrashidferrer", variant="proposed_plus")
    else:
        subprocess.run([sys.executable, str(builder), "--variant", "proposed_plus"], check=True)
    plus_metadata = tmp_path / "kaggle_dysample_plus_kernel" / "kernel-metadata.json"
    assert plus_metadata.is_file()
    assert json.loads(plus_metadata.read_text())["id"] == "dmrashidferrer/paddy-yolo-dysample-plus-r2000-fold0"
    assert (plain / "kernel-metadata.json").read_bytes() == original_metadata
