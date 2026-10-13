# Paper Table 5: five-fold configuration

The five-fold run is submitted only when explicitly requested. Its current
remote status and source checksum are recorded in `launch.json` after submission.

The supplied baseline paper, Section 2.1.5 (PDF page 4), uses a **3:1:1** fold
ratio: three partitions train, one validates, and one tests. Results are averaged
over five iterations. Table 5 (PDF page 5) specifies:

| Partition | Images | Instances |
| --- | ---: | ---: |
| 01 | 405 | 456 |
| 02 | 405 | 460 |
| 03 | 405 | 458 |
| 04 | 405 | 447 |
| 05 | 404 | 460 |
| Total | 2024 | 2281 |

The prepared partitions match both columns exactly. All original dataset splits
are pooled, retaining every image and annotation. An integer allocation keeps
the 12 exact-image duplicate groups together, meets the image/instance counts
as hard constraints, and balances class counts as a secondary objective.
It does not change any labels or bounding boxes. Seed 42 shuffles image groups
within equivalent allocation categories. The manifest records assignments,
checksums, library versions, class counts, and the method.

The paper does not publish image-level memberships or the precise order of the
validation/test rotation. These are reconstructed memberships matching its
counts. This implementation uses partition i as test, partition (i + 1) mod 5
as validation, and the other three as training. Therefore, matching Table 5's
counts does not establish that these are the authors' exact original splits.

| Iteration | Training images/instances | Validation images/instances | Test images/instances |
| --- | ---: | ---: | ---: |
| 0 | 1214 / 1365 | 405 / 460 | 405 / 456 |
| 1 | 1214 / 1363 | 405 / 458 | 405 / 460 |
| 2 | 1214 / 1376 | 405 / 447 | 405 / 458 |
| 3 | 1215 / 1374 | 404 / 460 | 405 / 447 |
| 4 | 1215 / 1365 | 405 / 456 | 404 / 460 |

Each image appears in training three times, validation once, and test once.
Exact image content never crosses split boundaries within any iteration.
All 16 classes are present in every train/validation/test subset. Validation
and test use the original images without added synthetic augmentation.

Local split YAMLs and lists are in
[`results/ema_dysample/paper_5fold/splits`](../../../results/ema_dysample/paper_5fold/splits).
They reference the existing audited local staging directory
`.local/kaggle-inspect/prepared`. The local absolute image paths are regenerated
inside Kaggle by the notebook; local YAMLs should not be uploaded as-is.

`run_config.json` retains the corrected RGG + EMA + DySample nano model,
`yolov8n.pt`, 300 epochs per fold, batch 32 across both GPUs (16 each),
AdamW `lr0=0.001`, zero bias warmup, and DIMIoU H. Each fold initializes a
fresh model. `run_folds.py` saves individual metrics/checkpoints/plots and
incremental mean/sample-standard-deviation summaries.

Build the prepared notebook from the repository root:

```powershell
.\.venv\Scripts\python.exe kaggle/ema_dysample/paper_5fold/build_notebook.py
```

Building generates `ema_dysample_paper_5fold.ipynb` and private GPU metadata.
It does not contact Kaggle or start training.


## Submitted run

[Kaggle notebook](https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-ema-dysample-paper-5fold), version 1, was submitted on 2026-10-13 20:13:39 Asia/Manila after user authorization. The session runs all five folds sequentially with a 7.5-hour limit. Current status is recorded in `launch.json`.


## Downloaded results

All five folds completed 300 epochs. The session ended with `CANCEL_ACKNOWLEDGED` during export; all 355 selected result artifacts were recovered and verified. The downloaded bundle is in [`results/ema_dysample/fivefold`](../../../results/ema_dysample/fivefold/README.md). Every best checkpoint loads locally and produces finite predictions.
