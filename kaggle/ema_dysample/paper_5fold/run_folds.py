"""Execute five prepared paper-protocol folds with independent initialization."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(__file__).with_name('run_config.json').read_text())
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory to preserve previous results')
    fold_paths = [args.data_dir / f'data_fold{index}.yaml' for index in range(5)]
    if not all(path.is_file() for path in fold_paths):
        raise FileNotFoundError('Prepare all five fold YAML configurations first')
    sys.path.insert(0, str(ROOT / 'kaggle/ema_dysample'))
    from runtime import run_logged
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for fold, data in enumerate(fold_paths):
        fold_output = output / f'fold{fold}'
        command = [sys.executable, str(ROOT / 'kaggle/ema_dysample/run_training.py'),
                   '--data', str(data.resolve()), '--output-dir', str(fold_output),
                   '--name', f'ema_dysample_paper_fold{fold}', '--hyp', str(ROOT / config['hyp']),
                   '--weights', config['initial_weights'], '--epochs', str(config['epochs_per_fold']),
                   '--batch', str(config['batch']), '--imgsz', str(config['imgsz']),
                   '--workers', str(config['workers']), '--device', config['device'],
                   '--patience', str(config['patience']), '--save-period', str(config['save_period'])]
        run_logged(command, fold_output / 'training.log', ROOT)
        report = json.loads((fold_output / 'metrics.json').read_text())
        report['fold'] = fold
        reports.append(report)
        summary = {'completed_folds': len(reports), 'folds': reports, 'aggregate': {}}
        for split in ('val', 'test'):
            summary['aggregate'][split] = {}
            for key in ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1'):
                scores = [row['metrics'][split][key] for row in reports]
                summary['aggregate'][split][key] = {'mean': statistics.mean(scores),
                                                    'std_sample': statistics.stdev(scores) if len(scores) > 1 else None}
        temporary = output / 'summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2) + '\n')
        temporary.replace(output / 'summary.json')


if __name__ == '__main__':
    main()
