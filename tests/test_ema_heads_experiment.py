"""Detection-branch EMA compatibility and reproducible five-fold launch."""
import ast
import base64
import csv
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
from paddy_yolo.modules.ema import EMA
from paddy_yolo.modules.dysample import DySample
from paddy_yolo.modules.rgg import RGG

CONFIG = 'configs/Paddy-YOLO-EMA-Heads.yaml'


def test_ema_receives_only_final_detection_features_and_backpropagates():
    # A wrong branch or concat index must fail the real forward pass or shape checks.
    assert Path(CONFIG).is_file(), 'Missing detection-branch EMA architecture'
    torch.set_num_threads(1)
    model = YOLO(CONFIG).model
    assert [i for i, m in enumerate(model.model) if isinstance(m, RGG)] == [2, 4, 6, 8]
    assert [i for i, m in enumerate(model.model) if isinstance(m, EMA)] == [22, 23, 24]
    assert not any(isinstance(m, DySample) for m in model.modules())
    assert [model.model[i].f for i in [22, 23, 24]] == [15, 18, 21]
    assert model.model[-1].f == [22, 23, 24]
    features = []
    hook = model.model[-1].register_forward_pre_hook(lambda m, args: features.extend(args[0]))
    model.eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 640, 640))
    hook.remove()
    assert [tuple(x.shape) for x in features] == [(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)]
    assert result[0].shape == (1, 20, 8400)
    model.train()
    outputs = model(torch.randn(2, 3, 64, 64))
    sum(x.square().mean() for x in outputs).backward()
    for i in [22, 23, 24]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[i].parameters())


def test_pretrained_detector_weights_survive_shifted_layer_index():
    import train
    initialize = getattr(train, 'initialize_pretrained_model', None)
    assert initialize is not None, 'Missing pretrained detection-head transfer'
    target = initialize(CONFIG, 'yolov8n.pt').model
    source = YOLO('yolov8n.pt').model
    assert torch.equal(target.model[0].conv.weight, source.model[0].conv.weight)
    assert torch.equal(target.model[-1].cv2[0][0].conv.weight, source.model[-1].cv2[0][0].conv.weight)
    assert torch.equal(target.model[18].cv1.conv.weight, source.model[18].cv1.conv.weight)
    assert target.model[-1].cv3[2][0].conv.weight.shape != source.model[-1].cv3[2][0].conv.weight.shape
    assert target.model[-1].nc == 16


def test_fresh_custom_weights_use_requested_seed_before_model_construction():
    from train import initialize_pretrained_model
    torch.manual_seed(11)
    first = initialize_pretrained_model(CONFIG, 'yolov8n.pt').model
    torch.manual_seed(999)
    second = initialize_pretrained_model(CONFIG, 'yolov8n.pt').model
    assert torch.equal(first.model[22].conv3x3.weight, second.model[22].conv3x3.weight)
    assert torch.equal(first.model[2].state_dict()['cv1.conv.weight'], second.model[2].state_dict()['cv1.conv.weight'])


def test_ema_fivefold_launches_fresh_models_and_preserves_reports(tmp_path, monkeypatch):
    import run_dysample_crossval as runner
    from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset
    from run_dysample_experiment import CONFIGS
    assert 'ema_heads' in CONFIGS, 'EMA variant is not registered for training'
    raw = tmp_path / 'raw'
    generate_dummy_dataset(raw, num_images=10, img_size=(64, 64), seed=42)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'T4 external boundary')
    calls = []

    def external_train(command, log_path):
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options)
        out = Path(options['--project']) / options['--name']
        (out / 'weights').mkdir(parents=True)
        (out / 'weights/best.pt').write_bytes(b'external training boundary')
        with (out / 'results.csv').open('w', newline='') as h:
            w = csv.writer(h)
            w.writerow(['epoch', 'val/box_loss'])
            w.writerows((i, 1.0) for i in range(1, 301))

    def external_eval(checkpoint, data, output, fold, variant='proposed'):
        assert variant == 'ema_heads'
        values = dict(precision=0.6, recall=0.5, mAP50=0.4, mAP50_95=0.3)
        return dict(overall=dict(fold=fold, class_id='', class_name='overall', images=2, instances=16, **values),
                    per_class=[dict(fold=fold, class_id=i, class_name=name, images=1, instances=1, **values)
                               for i, name in enumerate(runner.R2000_CLASSES)])

    monkeypatch.setattr(runner, 'run_training', external_train)
    monkeypatch.setattr(runner, 'evaluate_checkpoint', external_eval)
    output = tmp_path / 'ema_fivefold'
    runner.run(raw, output, variant='ema_heads')
    assert len(calls) == 5
    for i, options in enumerate(calls):
        assert options['--model'] == CONFIG and options['--weights'] == 'yolov8n.pt'
        assert options['--device'] == '0,1' and options['--batch'] == '32'
        assert options['--save-period'] == '50' and options['--epochs'] == '300'
        assert options['--name'] == f'fold{i}' and Path(options['--project']).name == 'ema_heads'
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed_folds'] == list(range(5))
    assert manifest['architecture'] == CONFIG and manifest['attention_indices'] == [22, 23, 24]
    assert len(list(csv.DictReader((output / 'per_class_and_overall.csv').open()))) == 85


def test_ema_notebook_packages_runnable_private_training(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('build_ema_heads_crossval_kernel'), 'Missing EMA notebook builder'
    from build_ema_heads_crossval_kernel import build
    output = build('dmrashidpferrer', tmp_path)
    metadata = json.loads((output / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'dmrashidpferrer/paddy-yolo-ema-heads-r2000-fivefold'
    assert metadata['is_private'] and metadata['enable_gpu']
    notebook = json.loads((output / metadata['code_file']).read_text())
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for cell in cells:
        compile(cell, 'ema-heads-notebook', 'exec')
    tree = ast.parse(cells[0])
    encoded = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'B64_ARCHIVE' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        assert CONFIG in archive.namelist()
        config = yaml.safe_load(archive.read(CONFIG))
        assert [row[2] for row in config['head']].count('EMA') == 3
        assert 'run_dysample_crossval.py' in archive.namelist()
    assert "'--variant', 'ema_heads'" in cells[-1]
    assert '/kaggle/working/ema_heads_fivefold' in cells[-1]


def test_ema_resume_rejects_dysample_checkpoint(tmp_path):
    from run_dysample_experiment import CONFIGS, validate_resume_checkpoint
    assert 'ema_heads' in CONFIGS, 'EMA resume variant is not registered'
    cfg = yaml.safe_load(Path('configs/Paddy-YOLO-DySample.yaml').read_text())
    checkpoint = tmp_path / 'last.pt'
    torch.save(dict(ema=SimpleNamespace(yaml=cfg)), checkpoint)
    with pytest.raises(ValueError, match='architecture'):
        validate_resume_checkpoint(checkpoint, 'ema_heads', 0)


def test_ema_resume_restores_saved_history_and_rejects_wrong_full_architecture(tmp_path):
    from run_dysample_crossval import validate_restored_resume
    config = yaml.safe_load(Path(CONFIG).read_text())
    settings = yaml.safe_load(Path('configs/hyp_paper.yaml').read_text())
    settings.update(seed=42, workers=2, name='fold0', project=str(tmp_path / 'ema_heads'),
                    data=str(tmp_path / 'dataset/data_fold0.yaml'))
    folder = tmp_path / 'ema_heads/fold0'
    (folder / 'weights').mkdir(parents=True)
    (folder / 'results.csv').write_text('epoch,val/box_loss\n1,1.0\n2,0.8\n3,0.6\n')
    saved = dict(ema=SimpleNamespace(yaml=config), train_args=settings, epoch=1,
                 optimizer={'state': {}}, scaler={}, best_fitness=0.5,
                 train_results={'epoch': [1, 2], 'val/box_loss': [1.0, 0.8]})
    checkpoint = folder / 'weights/epoch1.pt'
    torch.save(saved, checkpoint)
    torch.save(dict(train_metrics={'fitness': 0.5}), folder / 'weights/best.pt')
    restored = validate_restored_resume(checkpoint, tmp_path, 0, 'ema_heads')
    assert restored == [{'epoch': 1, 'val/box_loss': 1.0}, {'epoch': 2, 'val/box_loss': 0.8}]
    saved['ema'].yaml['head'][-1][0] = [15, 18, 21]
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='full architecture'):
        validate_restored_resume(checkpoint, tmp_path, 0, 'ema_heads')
