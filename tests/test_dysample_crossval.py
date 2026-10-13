"""Five independent DySample training folds and exact test-set reporting."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def payload(fold, value=0.5):
    from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES
    metrics = {"precision": value, "recall": value, "mAP50": value, "mAP50_95": value}
    overall = {"fold": fold, "class_id": "", "class_name": "overall", "images": 2, "instances": 16, **metrics}
    classes = [{"fold": fold, "class_id": i, "class_name": name, "images": 1, "instances": 1, **metrics}
               for i, name in enumerate(R2000_CLASSES)]
    return {"overall": overall, "per_class": classes}


def test_summary_requires_five_unique_folds_and_reports_sample_std(tmp_path):
    from run_dysample_crossval import aggregate_results
    results = [payload(i, value) for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    with pytest.raises(ValueError, match="five"):
        aggregate_results(results[:4], tmp_path)
    with pytest.raises(ValueError, match="five"):
        aggregate_results([results[0]] * 5, tmp_path)
    summary = aggregate_results(results, tmp_path)
    assert summary["mAP50"]["mean"] == pytest.approx(0.3)
    assert summary["mAP50"]["sample_std"] == pytest.approx(0.158113883)
    classes = list(csv.DictReader((tmp_path / "per_class_summary.csv").open()))
    assert len(classes) == 16
    assert float(classes[0]["mAP50_mean"]) == pytest.approx(0.3)
    assert int(classes[0]["evaluated_folds"]) == 5


def test_export_uses_class_ids_and_marks_absent_classes_unavailable():
    from run_dysample_crossval import export_metrics
    counts = [0] * 16
    counts[0], counts[2] = 2, 3
    metrics = SimpleNamespace(
        nt_per_class=counts, nt_per_image=counts,
        box=SimpleNamespace(mp=0.8, mr=0.6, map50=0.5, map=0.25, ap_class_index=[0, 2],
                            class_result=lambda i: [(0.8, 0.6, 0.5, 0.25), (0.4, 0.2, 0.3, 0.1)][i]),
    )
    result = export_metrics(metrics, fold=3, images_seen=4)
    assert result["overall"]["instances"] == 5
    assert result["overall"]["images"] == 4
    assert result["per_class"][2]["mAP50"] == 0.3
    assert result["per_class"][1]["mAP50"] is None
    metrics.box.map = float("nan")
    with pytest.raises(FloatingPointError):
        export_metrics(metrics, fold=3, images_seen=4)


def install_gpu_boundaries(monkeypatch, tmp_path, failure_fold=None):
    import run_dysample_crossval as runner
    from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset

    raw = tmp_path / "raw"
    generate_dummy_dataset(raw, num_images=10, img_size=(64, 64), seed=42)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda _: "T4 test boundary")
    calls = []

    def train(command, log_path):
        options = dict(zip(command[2::2], command[3::2]))
        fold = int(options["--name"].removeprefix("fold"))
        calls.append(options)
        if fold == failure_fold:
            raise RuntimeError("training worker failed")
        out = Path(options["--project"]) / options["--name"]
        (out / "weights").mkdir(parents=True, exist_ok=True)
        (out / "weights/best.pt").write_bytes(b"external trainer checkpoint")
        with (out / "results.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "val/box_loss"])
            writer.writerows((i, 1.0) for i in range(1, 301))

    monkeypatch.setattr(runner, "run_training", train)
    monkeypatch.setattr(runner, "evaluate_checkpoint", lambda checkpoint, data_yaml, output, fold: payload(fold))
    return runner, raw, calls


def test_fivefold_run_reinitializes_each_fold_and_saves_test_reports(tmp_path, monkeypatch):
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "fivefold"
    runner.run(raw, output)
    assert len(calls) == 5
    assert [c["--name"] for c in calls] == [f"fold{i}" for i in range(5)]
    for fold, command in enumerate(calls):
        assert command["--model"] == "configs/Paddy-YOLO-DySample.yaml"
        assert command["--weights"] == "yolov8n.pt"
        assert "--resume-from" not in command
        assert command["--batch"] == "32" and command["--device"] == "0,1"
        assert command["--epochs"] == "300" and command["--save-period"] == "50"
        assert Path(command["--data"]).name == f"data_fold{fold}.yaml"
        assert json.loads((output / f"fold{fold}_test_metrics.json").read_text())["history_check"]["all_values_finite"]
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["completed_folds"] == [0, 1, 2, 3, 4]
    assert len(list(csv.DictReader((output / "individual_folds.csv").open()))) == 5
    assert len(list(csv.DictReader((output / "per_class_and_overall.csv").open()))) == 85
    assert (output / "fivefold_summary.json").exists()


def test_failed_fold_keeps_completed_results_without_final_summary(tmp_path, monkeypatch):
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path, failure_fold=1)
    output = tmp_path / "fivefold"
    with pytest.raises(RuntimeError, match="training worker failed"):
        runner.run(raw, output)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "error" and manifest["completed_folds"] == [0]
    assert (output / "fold0_test_metrics.json").exists()
    assert not (output / "fivefold_summary.json").exists()


def test_new_fivefold_kernel_packages_plain_model_and_reporting_runner(tmp_path):
    import ast
    import base64
    import io
    import zipfile
    import yaml
    from build_dysample_crossval_kernel import build

    output = build("dmrashidferrer", tmp_path)
    metadata = json.loads((output / "kernel-metadata.json").read_text())
    assert metadata["id"] == "dmrashidferrer/paddy-yolo-dysample-r2000-fivefold"
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True
    notebook = json.loads((output / metadata["code_file"]).read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in cells:
        compile(cell, "fivefold-cell", "exec")
    tree = ast.parse(cells[0])
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "B64_ARCHIVE" for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        assert "run_dysample_crossval.py" in archive.namelist()
        assert "run_baseline_crossval.py" in archive.namelist()
        cfg = yaml.safe_load(archive.read("configs/Paddy-YOLO-DySample.yaml"))
        assert [cfg["head"][i][3] for i in (0, 3)] == [[2, "lp", 4, False]] * 2
    assert "run_dysample_crossval.py" in cells[1]
    assert "run_dysample_experiment.py" not in cells[1]


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("original", [None, "0,1"])
def test_test_evaluation_preserves_gpu_visibility_for_next_fold(tmp_path, monkeypatch, fails, original):
    import os
    import run_dysample_crossval as runner
    from paddy_yolo.modules.dysample import DySample
    if original is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", original)
    monkeypatch.setattr(runner, "YOLO", lambda _: SimpleNamespace(
        model=SimpleNamespace(model={10: DySample(256), 13: DySample(128)})))

    class Validator:
        def __init__(self, save_dir, args):
            assert args["split"] == "test" and args["half"] is False
            self.metrics, self.seen = None, 2

        def __call__(self, model):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if fails:
                raise RuntimeError("evaluation failed")

    monkeypatch.setattr(runner, "FP32DetectionValidator", Validator)
    monkeypatch.setattr(runner, "export_metrics", lambda *args: payload(0))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    if fails:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            runner.evaluate_checkpoint("best.pt", "fold0.yaml", tmp_path, 0)
    else:
        runner.evaluate_checkpoint("best.pt", "fold0.yaml", tmp_path, 0)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == original


def test_continue_completed_folds_preserves_reports_and_restarts_remaining_folds(tmp_path, monkeypatch):
    runner, raw, _ = install_gpu_boundaries(monkeypatch, tmp_path, failure_fold=2)
    output = tmp_path / "fivefold"
    with pytest.raises(RuntimeError):
        runner.run(raw, output)
    original_fold0 = (output / "fold0_test_metrics.json").read_bytes()
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path)
    runner.run(raw, output, start_fold=2)
    assert [c["--name"] for c in calls] == ["fold2", "fold3", "fold4"]
    assert all(c["--weights"] == "yolov8n.pt" for c in calls)
    assert (output / "fold0_test_metrics.json").read_bytes() == original_fold0
    assert json.loads((output / "run_manifest.json").read_text())["completed_folds"] == list(range(5))


def restored_resume_fixture(output, fold=1):
    import torch
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/Paddy-YOLO-DySample.yaml").read_text())
    settings = yaml.safe_load((root / "configs/hyp_paper.yaml").read_text())
    settings.update(seed=42, workers=2, name=f"fold{fold}", project=str(output / "proposed"),
                    data=str(output / "dataset" / f"data_fold{fold}.yaml"))
    run_dir = output / "proposed" / f"fold{fold}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "val/box_loss"])
        writer.writerows((i, 1.0) for i in range(1, 5))
    saved = {"ema": SimpleNamespace(yaml=config), "train_args": settings, "epoch": 2,
             "optimizer": {"state": {}}, "scaler": {}, "best_fitness": 0.5,
             "train_results": {"epoch": [1, 2, 3], "val/box_loss": [1.0, 1.0, 1.0]}}
    checkpoint = run_dir / "weights/last.pt"
    torch.save(saved, checkpoint)
    torch.save({"train_args": settings, "train_metrics": {"fitness": 0.5}}, run_dir / "weights/best.pt")
    return checkpoint, saved


@pytest.mark.parametrize("mutation", ["learning_rate", "architecture", "history", "missing_history"])
def test_resume_rejects_incompatible_or_incomplete_restored_state(tmp_path, monkeypatch, mutation):
    import torch
    runner, raw, _ = install_gpu_boundaries(monkeypatch, tmp_path, failure_fold=1)
    output = tmp_path / "fivefold"
    with pytest.raises(RuntimeError):
        runner.run(raw, output)
    checkpoint, saved = restored_resume_fixture(output)
    if mutation == "learning_rate":
        saved["train_args"]["lr0"] = 0.8
    elif mutation == "architecture":
        saved["ema"].yaml["backbone"][0][3][0] = 128
    elif mutation == "history":
        saved["train_results"]["val/box_loss"][0] = 8.0
    else:
        (checkpoint.parent.parent / "results.csv").unlink()
    torch.save(saved, checkpoint)
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        runner.run(raw, output, start_fold=1, resume_from=checkpoint)
    assert calls == []


def test_resume_restores_checkpoint_history_and_only_resumes_current_fold(tmp_path, monkeypatch):
    runner, raw, _ = install_gpu_boundaries(monkeypatch, tmp_path, failure_fold=1)
    output = tmp_path / "fivefold"
    with pytest.raises(RuntimeError):
        runner.run(raw, output)
    checkpoint, _ = restored_resume_fixture(output)
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path)
    original_train = runner.run_training

    def train(command, log):
        if "--resume-from" in command:
            restored = list(csv.DictReader((checkpoint.parent.parent / "results.csv").open()))
            assert [int(r["epoch"]) for r in restored] == [1, 2, 3]
        original_train(command, log)

    monkeypatch.setattr(runner, "run_training", train)
    runner.run(raw, output, start_fold=1, resume_from=checkpoint)
    assert calls[0]["--resume-from"] == str(checkpoint)
    assert "--weights" not in calls[0]
    assert all(c["--weights"] == "yolov8n.pt" for c in calls[1:])


def test_completed_training_can_retry_failed_test_without_retraining(tmp_path, monkeypatch):
    runner, raw, _ = install_gpu_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "fivefold"
    evaluate = runner.evaluate_checkpoint
    monkeypatch.setattr(runner, "evaluate_checkpoint", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test failed")))
    with pytest.raises(RuntimeError, match="test failed"):
        runner.run(raw, output)
    runner, raw, calls = install_gpu_boundaries(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "evaluate_checkpoint", evaluate)
    runner.run(raw, output, start_fold=0)
    assert [c["--name"] for c in calls] == ["fold1", "fold2", "fold3", "fold4"]
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "complete"
