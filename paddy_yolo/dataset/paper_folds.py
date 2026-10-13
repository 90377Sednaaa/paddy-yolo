"""Reconstruct Table 5 R2000 counts with duplicate-safe 3:1:1 rotation.

The paper provides counts, not image-level fold membership. These are newly
generated folds matching those counts; original author membership is unknown.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

TABLE5_IMAGES = (405, 405, 405, 405, 404)
TABLE5_INSTANCES = (456, 460, 458, 447, 460)


def assign_table5_partitions(records, seed=42, image_targets=TABLE5_IMAGES,
                             instance_targets=TABLE5_INSTANCES):
    """Assign whole image-content groups to exact image and annotation targets.

    Integer allocation operates on categories of identical group sizes and
    class-count vectors, keeping the optimization small. Absolute per-class
    deviations from one fifth of the dataset provide a secondary balancing
    objective. Image and instance counts are hard constraints.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    if len(image_targets) != 5 or len(instance_targets) != 5:
        raise ValueError('Exactly five image and instance targets are required')
    if any(value <= 0 for value in (*image_targets, *instance_targets)):
        raise ValueError('Fold targets must be positive')
    if len(records) != sum(image_targets) or sum(sum(row['class_boxes']) for row in records) != sum(instance_targets):
        raise ValueError('Dataset totals do not match the requested image and instance targets')
    if not records or any(len(row['class_boxes']) != 16 for row in records):
        raise ValueError('Every image must provide counts for the 16 classes')
    grouped = defaultdict(list)
    for index, row in enumerate(records):
        grouped[row['image_sha256']].append(index)
    if len(grouped) < 5:
        raise ValueError('At least five distinct image groups are required')
    categories = defaultdict(list)
    for indices in grouped.values():
        counts = tuple(sum(records[index]['class_boxes'][class_id] for index in indices) for class_id in range(16))
        categories[(len(indices), counts)].append(indices)
    signatures = sorted(categories)
    category_count = len(signatures)
    allocation_count = category_count * 5
    # Two nonnegative slack variables express each class deviation per fold.
    variable_count = allocation_count + 5 * 16 * 2
    objective = np.zeros(variable_count)
    upper_bounds = np.full(variable_count, np.inf)
    for category, signature in enumerate(signatures):
        upper_bounds[category * 5:(category + 1) * 5] = len(categories[signature])
    class_totals = [sum(row['class_boxes'][class_id] for row in records) for class_id in range(16)]
    rows, columns, values, lower, upper = [], [], [], [], []

    def constraint(coefficients, target):
        row = len(lower)
        for column, value in coefficients:
            if value:
                rows.append(row)
                columns.append(column)
                values.append(value)
        lower.append(target)
        upper.append(target)

    for category, signature in enumerate(signatures):
        constraint([(category * 5 + fold, 1) for fold in range(5)], len(categories[signature]))
    for fold in range(5):
        constraint([(category * 5 + fold, signature[0]) for category, signature in enumerate(signatures)], image_targets[fold])
        constraint([(category * 5 + fold, sum(signature[1])) for category, signature in enumerate(signatures)], instance_targets[fold])
        for class_id in range(16):
            positive = allocation_count + (fold * 16 + class_id) * 2
            negative = positive + 1
            objective[positive:negative + 1] = 1 / max(class_totals[class_id], 1)
            coefficients = [(category * 5 + fold, signature[1][class_id]) for category, signature in enumerate(signatures)]
            constraint(coefficients + [(positive, -1), (negative, 1)], class_totals[class_id] / 5)
    matrix = coo_matrix((values, (rows, columns)), shape=(len(lower), variable_count)).tocsc()
    integrality = np.zeros(variable_count)
    integrality[:allocation_count] = 1
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(np.zeros(variable_count), upper_bounds),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': 60, 'mip_rel_gap': 0.01})
    if result.x is None:
        raise ValueError('Cannot match the target counts while keeping duplicate images together')
    allocation = np.rint(result.x[:allocation_count]).astype(int).reshape(category_count, 5)
    partitions = [-1] * len(records)
    rng = np.random.default_rng(seed)
    for category, signature in enumerate(signatures):
        groups = categories[signature]
        shuffled = rng.permutation(len(groups))
        start = 0
        for fold, count in enumerate(allocation[category]):
            for group_index in shuffled[start:start + count]:
                for image_index in groups[group_index]:
                    partitions[image_index] = fold
            start += count
        if start != len(groups):
            raise ValueError('Invalid integer group allocation')
    if -1 in partitions:
        raise ValueError('Unassigned images remain')
    for fold in range(5):
        subset = [row for row, partition in zip(records, partitions) if partition == fold]
        if len(subset) != image_targets[fold] or sum(sum(row['class_boxes']) for row in subset) != instance_targets[fold]:
            raise ValueError('Integer allocation failed exact count verification')
    return partitions


def prepare_paper_folds(staged_config: Path, output_dir: Path, seed=42):
    """Use an audited, writable staging of all original dataset splits.

    Input is the data.yaml produced by kaggle/baseline/prepare_dataset.py.
    Preserve paths through staging rather than resolving image symlinks, so
    Ultralytics finds the copied labels and can write its dataset caches.
    """
    import scipy

    staged_config = Path(staged_config)
    source = staged_config.parent.absolute()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty split-output directory')
    audit_path = source / 'dataset_audit.json'
    if not audit_path.is_file():
        raise ValueError('Stage and audit the dataset before generating paper folds')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['dataset'] != 'dmrashidpferrer/r2000-pestnet' or audit['cross_split_duplicates']:
        raise ValueError('Expected the audited R2000 dataset without original split leakage')
    config = yaml.safe_load(staged_config.read_text(encoding='utf-8'))
    names = config['names']
    if config.get('nc') != 16 or len(names) != 16:
        raise ValueError('Expected 16 R2000 classes')
    records = []
    for split in ('train', 'val', 'test'):
        for image in sorted((source / 'images' / split).rglob('*'), key=lambda path: path.as_posix()):
            if not image.is_file() or image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                continue
            subpath = image.relative_to(source / 'images' / split)
            label = source / 'labels' / split / subpath.with_suffix('.txt')
            counts = [0] * 16
            for line in label.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
            records.append({'image': image.relative_to(source).as_posix(), 'original_split': split,
                            'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                            'label_sha256': hashlib.sha256(label.read_bytes()).hexdigest(), 'class_boxes': counts})
    partitions = assign_table5_partitions(records, seed)
    for row, partition in zip(records, partitions):
        row['partition'] = partition
    manifest = {'dataset': audit['dataset'], 'protocol': 'paper_table5_three_train_one_val_one_test',
                'paper_reference': 'Section 2.1.5 (PDF page 4), Table 5 (PDF page 5)',
                'membership': 'new allocation matching published counts; author image lists unavailable',
                'seed': seed, 'images': len(records), 'instances': sum(sum(row['class_boxes']) for row in records),
                'unique_image_groups': len({row['image_sha256'] for row in records}),
                'grouping': 'image_sha256', 'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
                'partition_counts': [{'partition': index, 'images': TABLE5_IMAGES[index], 'instances': TABLE5_INSTANCES[index]}
                                     for index in range(5)], 'records': records, 'iterations': []}
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for fold in range(5):
        split_partitions = {'test': {fold}, 'val': {(fold + 1) % 5},
                            'train': set(range(5)) - {fold, (fold + 1) % 5}}
        fold_config = {'path': output.as_posix(), 'nc': 16, 'names': names}
        iteration = {'fold': fold, 'splits': {}}
        hash_sets = {}
        for split, included in split_partitions.items():
            subset = [row for row in records if row['partition'] in included]
            counts = [sum(row['class_boxes'][class_id] for row in subset) for class_id in range(16)]
            if any(count == 0 for count in counts):
                raise ValueError(f'Fold {fold} {split} lacks an annotated class')
            hash_sets[split] = {row['image_sha256'] for row in subset}
            filename = f'fold{fold}_{split}.txt'
            (output / filename).write_text('\n'.join((source / row['image']).as_posix() for row in subset) + '\n', encoding='utf-8')
            fold_config[split] = filename
            iteration['splits'][split] = {'partitions': sorted(included), 'images': len(subset),
                                         'instances': sum(counts), 'class_boxes': counts}
        if any(hash_sets[first] & hash_sets[second] for first, second in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
            raise ValueError('Duplicate content crosses a training, validation, or test boundary')
        path = output / f'data_fold{fold}.yaml'
        path.write_text(yaml.safe_dump(fold_config, sort_keys=False), encoding='utf-8')
        paths.append(path)
        manifest['iterations'].append(iteration)
    (output / 'fold_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return paths
