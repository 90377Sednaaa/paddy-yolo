"""P4-to-P3-only DySample training and held-out evaluation."""
import ast
import base64
import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch
import yaml
from ultralytics import YOLO

import paddy_yolo
from paddy_yolo.modules.dysample import DySample

CONFIG = 'configs/Paddy-YOLO-DySample-P3.yaml'


def test_only_p3_upsampling_changes_and_detection_features_backpropagate():
    assert Path(CONFIG).is_file(), 'Missing P3-only configuration'
    base = yaml.safe_load(Path('configs/paddy-yolo.yaml').read_text())
    proposed = yaml.safe_load(Path(CONFIG).read_text())
    assert proposed['backbone'] == base['backbone']
    assert [(i, row) for i, row in enumerate(proposed['head']) if row != base['head'][i]] == [
        (3, [-1, 1, 'DySample', [2, 'lp', 4, False]])]
    torch.set_num_threads(1)
    model = YOLO(CONFIG).model
    assert [i for i, layer in enumerate(model.model) if isinstance(layer, DySample)] == [13]
    assert isinstance(model.model[10], torch.nn.Upsample) and model.model[10].mode == 'nearest'
    assert model.model[13].offset.in_channels == 128
    shapes = []
    hook = model.model[-1].register_forward_pre_hook(
        lambda m, args: shapes.append([tuple(x.shape) for x in args[0]]))
    model.eval()
    with torch.no_grad():
        pred = model(torch.randn(1, 3, 640, 640))
    hook.remove()
    assert pred[0].shape == (1, 20, 8400)
    assert shapes == [[(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)]]
    model.train()
    result = model(torch.randn(2, 3, 64, 64))
    sum(x.square().mean() for x in result).backward()
    grad = model.model[13].offset.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_p3_training_command_keeps_settings_and_resume_excludes_initial_weights(tmp_path):
    from run_dysample_experiment import CONFIGS, training_command
    assert 'p3_only' in CONFIGS, 'P3-only variant is not registered'
    command = training_command('p3_only', 0, 'data_fold0.yaml', tmp_path, 2)
    options = dict(zip(command[2::2], command[3::2]))
    assert options['--model'] == CONFIG and options['--weights'] == 'yolov8n.pt'
    assert options['--batch'] == '32' and options['--device'] == '0,1'
    assert options['--epochs'] == '300' and options['--save-period'] == '50'
    assert Path(options['--project']).name == 'p3_only'
    resumed = training_command('p3_only', 0, 'data_fold0.yaml', tmp_path, 2, 'last.pt')
    assert '--resume-from' in resumed and '--weights' not in resumed


def test_p3_full_notebook_has_unique_identity_and_packages_model(tmp_path):
    from run_dysample_experiment import CONFIGS
    assert 'p3_only' in CONFIGS, 'P3-only launch support missing'
    from build_dysample_full_kernel import build
    output = build('dmrashidferrer', 0, tmp_path, variant='p3_only')
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'dmrashidferrer/paddy-yolo-dysample-p3-only-r2000-fold0'
    assert metadata['is_private'] and metadata['enable_gpu']
    nb = json.loads((output / metadata['code_file']).read_text())
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    for cell in cells:
        compile(cell, 'p3-only-notebook', 'exec')
    tree = ast.parse(cells[0])
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'B64_ARCHIVE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        assert CONFIG in archive.namelist() and 'run_dysample_p3_experiment.py' in archive.namelist()
        assert [r[2] for r in yaml.safe_load(archive.read(CONFIG))['head']].count('DySample') == 1
    assert 'run_dysample_p3_experiment.py' in cells[-1]
    assert "VARIANT = 'p3_only'" in cells[-1] and 'FOLD = 0' in cells[-1]


@pytest.mark.parametrize('training_fails', [False, True])
def test_runner_trains_one_fold_and_preserves_exact_test_reports(tmp_path, monkeypatch, training_fails):
    assert importlib.util.find_spec('run_dysample_p3_experiment'), 'Missing single-fold reporting runner'
    import run_dysample_p3_experiment as runner
    from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset, R2000_CLASSES
    raw = generate_dummy_dataset(tmp_path / 'raw', num_images=10, img_size=(64, 64), seed=42)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'T4 external boundary')
    calls = []

    def external_train(command, log):
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options)
        if training_fails:
            raise RuntimeError('training process failed')
        folder = Path(options['--project']) / options['--name']
        (folder / 'weights').mkdir(parents=True)
        (folder / 'weights/best.pt').write_bytes(b'external training boundary')
        with (folder / 'results.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['epoch', 'val/box_loss'])
            writer.writerows((i, 1.0) for i in range(1, 301))

    def external_eval(checkpoint, data, output, fold, variant):
        assert variant == 'p3_only' and fold == 0
        values = dict(precision=0.6, recall=0.5, mAP50=0.4, mAP50_95=0.3)
        return dict(overall=dict(fold=0, class_id='', class_name='overall', images=2, instances=16, **values),
                    per_class=[dict(fold=0, class_id=i, class_name=name, images=1, instances=1, **values)
                               for i, name in enumerate(R2000_CLASSES)])

    monkeypatch.setattr(runner, 'run_training', external_train)
    monkeypatch.setattr(runner, 'evaluate_checkpoint', external_eval)
    output = tmp_path / 'p3_output'
    if training_fails:
        with pytest.raises(RuntimeError, match='training process failed'):
            runner.run(raw, output, fold=0)
        assert not (output / 'fold0_test_metrics.json').exists()
    else:
        runner.run(raw, output, fold=0)
        assert len(list(csv.DictReader((output / 'per_class_and_overall.csv').open()))) == 17
        report = json.loads((output / 'fold0_test_metrics.json').read_text())
        assert report['overall']['mAP50'] == 0.4 and report['history_check']['epochs'] == 300
    assert len(calls) == 1 and calls[0]['--weights'] == 'yolov8n.pt'
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['status'] == ('error' if training_fails else 'complete')
    assert manifest['completed_folds'] == ([] if training_fails else [0])
    assert manifest['dysample_indices'] == [13]
    assert not (output / 'fivefold_summary.json').exists()


def test_p3_evaluation_rejects_two_stage_checkpoint_before_validation(tmp_path):
    from run_dysample_experiment import CONFIGS
    assert 'p3_only' in CONFIGS, 'P3-only evaluation support missing'
    from run_dysample_crossval import evaluate_checkpoint
    model = YOLO('configs/Paddy-YOLO-DySample.yaml')
    checkpoint = tmp_path / 'wrong.pt'
    model.save(checkpoint)
    with pytest.raises(ValueError, match='P3-only'):
        evaluate_checkpoint(checkpoint, 'unused.yaml', tmp_path, 0, device='cpu', variant='p3_only')
