"""Exact paper image/instance targets and content-group integrity."""
from collections import Counter

import pytest

from paddy_yolo.dataset.paper_folds import assign_table5_partitions, TABLE5_IMAGES, TABLE5_INSTANCES


def records():
    result = []
    for index in range(80):
        counts = [0] * 16
        counts[index % 16] = 2 if index in (0, 1) else 1
        result.append({'image_sha256': str(index), 'class_boxes': counts})
    counts = [0] * 16
    counts[0] = 1
    result.append({'image_sha256': '0', 'class_boxes': counts})
    return result


def test_exact_images_instances_and_duplicate_grouping():
    data = records()
    images = (17, 16, 16, 16, 16)
    boxes = (19, 16, 16, 16, 16)
    partitions = assign_table5_partitions(data, image_targets=images, instance_targets=boxes)
    assert partitions[0] == partitions[80]
    for fold in range(5):
        subset = [row for row, partition in zip(data, partitions) if partition == fold]
        assert len(subset) == images[fold]
        assert sum(sum(row['class_boxes']) for row in subset) == boxes[fold]
    visits = {split: Counter() for split in ('train', 'val', 'test')}
    for fold in range(5):
        for index, partition in enumerate(partitions):
            split = 'test' if partition == fold else 'val' if partition == (fold + 1) % 5 else 'train'
            visits[split][index] += 1
    for split, count in (('train', 3), ('val', 1), ('test', 1)):
        assert len(visits[split]) == len(data)
        assert set(visits[split].values()) == {count}


def test_reconstructed_membership_is_reproducible():
    kwargs = {'image_targets': (17, 16, 16, 16, 16), 'instance_targets': (19, 16, 16, 16, 16)}
    assert assign_table5_partitions(records(), seed=42, **kwargs) == assign_table5_partitions(records(), seed=42, **kwargs)


def test_rejects_dataset_totals_that_do_not_match_table5():
    assert sum(TABLE5_IMAGES) == 2024 and sum(TABLE5_INSTANCES) == 2281
    with pytest.raises(ValueError, match='totals'):
        assign_table5_partitions(records())


def test_impossible_targets_do_not_split_duplicate_images():
    counts = [1] + [0] * 15
    data = [{'image_sha256': 'duplicate' if index < 4 else str(index), 'class_boxes': counts} for index in range(8)]
    with pytest.raises(ValueError, match='keeping duplicate'):
        assign_table5_partitions(data, image_targets=(2, 2, 2, 1, 1), instance_targets=(2, 2, 2, 1, 1))
