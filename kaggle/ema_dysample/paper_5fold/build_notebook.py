"""Prepare a private Kaggle notebook following Table 5; never submit it."""
import base64
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'kaggle/baseline'))
from build_notebook import cell


def build():
    config = json.loads((OUTPUT / 'run_config.json').read_text())
    sources = [ROOT / name for name in ('train.py', 'val.py', 'predict.py', 'verify_paddy_yolo.py',
                                        'requirements.txt', 'pytest.ini', 'RGG_EMA_DySample_changes.txt')]
    sources += sorted((ROOT / 'paddy_yolo').rglob('*.py'))
    sources += sorted((ROOT / 'tests').glob('test_*.py'))
    sources += [ROOT / 'configs' / name for name in ('paddy-yolo.yaml', 'paddy-yolo-ema-dysample.yaml',
                                                    'hyp_paper.yaml', 'hyp_ema_dysample.yaml', 'r2000.yaml')]
    sources += [ROOT / 'kaggle/baseline' / name for name in ('prepare_dataset.py', 'run_baseline.py', 'requirements.txt')]
    sources += [ROOT / 'kaggle/ema_dysample' / name for name in ('run_training.py', 'runtime.py', 'requirements.txt')]
    sources += [OUTPUT / name for name in ('run_config.json', 'run_folds.py', 'requirements.txt')]
    sources += [ROOT / 'third_party/dysample-LICENSE.txt']
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            content = source.read_bytes()
            if re.search(rb'KGAT_[A-Za-z0-9_\-]{16,}', content):
                raise ValueError('Credentials must not be included in the notebook')
            archive.writestr(source.relative_to(ROOT).as_posix(), content)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}}, 'cells': [
        cell('''
            # R2000: paper Table 5, five independent EMA + DySample models

            Three partitions train, one validates, and one tests (**3:1:1**).
            Exact partition counts (images/instances): 405/456, 405/460, 405/458,
            405/447, 404/460. All 2024 images and 2281 instances are retained.
            The paper does not publish image memberships; these are reconstructed
            assignments matching its counts, with exact duplicates kept together.
            All 16 classes must appear in each train/validation/test subset.

            Corrected RGG + EMA backbone and DySample neck; nano, `yolov8n.pt`,
            300 epochs per fold, AdamW 0.001, zero bias warmup, batch 32 across
            both T4 GPUs (16 each). Five models initialize independently.
            Each best checkpoint is selected using validation, then tested.
            Results are averaged across folds. This notebook executes all five
            independent folds sequentially in one GPU session.
        ''', 'markdown'),
        cell(f'''
            import base64, io, os, sys, zipfile
            from pathlib import Path
            PROJECT = Path('/kaggle/working/paddy_yolo_paper_source')
            PROJECT.mkdir(parents=True, exist_ok=True)
            B64_SOURCE = '{encoded}'
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(B64_SOURCE))) as archive:
                archive.extractall(PROJECT)
            os.chdir(PROJECT)
            sys.path.insert(0, str(PROJECT))
            os.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/paper_settings/ultralytics'
            os.environ['MPLCONFIGDIR'] = '/kaggle/working/paper_settings/matplotlib'
            os.environ['WANDB_MODE'] = 'disabled'
            os.environ['PYTHONUNBUFFERED'] = '1'
        '''),
        cell('''
            import json
            RUN = json.loads((PROJECT / 'kaggle/ema_dysample/paper_5fold/run_config.json').read_text())
            print(json.dumps(RUN, indent=2))
            sys.path.insert(0, str(PROJECT / 'kaggle/ema_dysample'))
            from runtime import configure_ultralytics, run_logged
            run_logged([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                        str(PROJECT / 'kaggle/ema_dysample/paper_5fold/requirements.txt')],
                       Path('/kaggle/working/environment_setup.log'), PROJECT)
            import torch, ultralytics
            assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
            assert ultralytics.__version__ == '8.3.18'
            configure_ultralytics()
            run_logged([sys.executable, '-m', 'pytest', '-q', '--tb=short'],
                       Path('/kaggle/working/preflight_tests.log'), PROJECT)
        '''),
        cell('''
            sys.path.insert(0, str(PROJECT / 'kaggle/baseline'))
            from prepare_dataset import prepare_dataset
            from paddy_yolo.dataset.paper_folds import prepare_paper_folds
            STAGED = prepare_dataset(Path('/kaggle/input'), Path('/kaggle/working/paper_original_split'))
            DATA = Path('/kaggle/working/paper_5fold_splits')
            prepare_paper_folds(STAGED, DATA, seed=RUN['seed'])
            manifest = json.loads((DATA / 'fold_manifest.json').read_text())
            print(json.dumps(manifest['partition_counts'], indent=2))
            from ultralytics import YOLO
            pretrained = YOLO(RUN['initial_weights'])
            del pretrained
        '''),
        cell('''
            OUTPUT = Path('/kaggle/working/ema_dysample_paper_5fold_output')
            run_logged([sys.executable, str(PROJECT / 'kaggle/ema_dysample/paper_5fold/run_folds.py'),
                        '--data-dir', str(DATA), '--output-dir', str(OUTPUT)],
                       Path('/kaggle/working/paper_training.log'), PROJECT)
            print((OUTPUT / 'summary.json').read_text())
        '''),
    ]}
    for index, entry in enumerate(notebook['cells']):
        entry['id'] = f'paper-folds-{index:02d}'
    path = OUTPUT / 'ema_dysample_paper_5fold.ipynb'
    path.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    metadata = {'id': f'{config["account"]}/{config["kernel_slug"]}', 'title': config['title'],
                'code_file': path.name, 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': [config['dataset']], 'competition_sources': [], 'kernel_sources': []}
    (OUTPUT / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared notebook: {len(sources)} source files. No Kaggle submission performed.')


if __name__ == '__main__':
    build()
