# Baseline results and corrective rerun

Source: version 1 of the [private baseline notebook](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-baseline-fivefold-r2000-20261011),
retrieved October 12, 2026. Comparison uses Tables 8 and 13 of the supplied paper.
Original artifacts are preserved under ignored `data/baseline_crossval_summary/`.

| Metric | Paper baseline | First run, mean ± sample SD | Difference in percentage points |
| --- | ---: | ---: | ---: |
| Precision | 67.9% | 59.93% ± 2.95 | -7.97 |
| Recall | 53.8% | 52.95% ± 4.51 | -0.85 |
| F1 | 60.0% | 56.13% ± 3.24 | -3.87 |
| mAP@50 | 61.1% | 57.03% ± 2.84 | -4.07 |
| mAP@50:95 | 43.6% | 39.99% ± 3.05 | -3.61 |

The means and sample standard deviations were independently recomputed from
`individual_folds.csv` and exactly match `fivefold_summary.json`. F1 is the
harmonic mean of each fold's reported macro precision and recall, then averaged
over the five folds. This is the convention used by the run; it is not mean
per-class F1. The baseline model has 3,013,968 training parameters and 8.2 GFLOPs,
which round to Table 13's 3.01M / 8.2G values.

## Verified settings and split checks

All five `results.csv` files contain exactly 300 sequential epochs. All saved
arguments specify batch 32, input size 640, AdamW, lr0 0.01, lrf 0.01, momentum
0.937, weight decay 0.0005 and disabled early stopping. Logs confirm AdamW used
these values, rather than automatic optimizer selection. Training started from
`yolov8n.yaml`, with no transferred checkpoint. The legacy `pretrained: true`
argument did not load weights for this YAML initialization; the rerun records
`pretrained: false` explicitly.

The dataset has 2,024 images and 2,281 boxes. No corrupt images were reported.
Fold train/validation/test counts are 1214/405/405, 1214/405/405,
1214/405/405, 1215/404/405 and 1215/405/404. Path sets are disjoint within each
iteration; each sample appears in test once, validation once and training three
times. The splitter keeps byte-identical images together. Authors' exact fold
membership remains unavailable.

## Verified numerical failure

Every fold recorded non-finite **validation** losses in early epochs:

| Fold | Non-finite validation epochs |
| --- | --- |
| 0 | 1–5: classification loss Inf |
| 1 | 1–5 and 9: classification loss Inf |
| 2 | 1, 3–5: classification loss Inf; 2: box/classification/DFL losses NaN |
| 3 | 1–5: classification loss Inf |
| 4 | 1–5: classification loss Inf |

All recorded training losses and final test metrics are finite. Therefore these
are real saved test measurements, and the lower score alone does not establish
an implementation error. They do, however, come from a run with unreliable
early validation losses. The accuracy gap cannot be attributed solely to this
problem without the corrected run.

The pinned Ultralytics validator forces FP16 validation whenever training AMP
is on, and calculates losses outside autocast. FP16 reductions can overflow.
The inherited bias warmup also starts at 0.1; the first epoch actually logged
bias LR 0.0707895, over seven times the reported base LR 0.01. The optimizer's
own automatic-selection path sets this warmup to zero for Adam-family training.
These observations support the following stability corrections; the paper does
not specify warmup or numerical precision, so these remain disclosed choices.

## Corrections and rerun

- Keep standard YOLOv8-N architecture, CIoU, BCE, DFL, AMP training, batch 32,
  640-pixel inputs, AdamW and 300 epochs.
- Evaluate during training in FP32, restoring the training AMP flag afterward.
- Use AdamW-compatible `warmup_bias_lr: 0.0` to remove the excessive initial bias LR.
- Fail on non-finite training or validation values at epoch end.
- Require a complete, finite 300-epoch CSV before evaluating/publishing a fold.
- Save membership hashes to verify identical folds across experiments.

All five folds are rerun from scratch using the same dataset, seed and splitting
algorithm. Version 1 outputs are retained, and are not replaced with estimates or
hand-edited metrics. PyTorch/CUDA differ from the paper's PyTorch 2.0.1 / CUDA 11.8;
the first run used PyTorch 2.11.0+cu128, Ultralytics 8.3.253 and a Tesla T4.
The final LR convention and other replication assumptions remain documented in
[the original audit](replication-audit.md).

The CIoU corrective rerun was submitted as version 2 and confirmed running.
The user then clarified that the intended model is **RGG + IMPDIoU**. Its separate
[verified experiment](rgg-impdiou-run.md) uses ratio 1.3 and compares against
Table 13's `+ RGG + IMPD` row (69.7% mAP@50, 50.8% mAP@50:95).
This report remains the comparison of the original CIoU baseline only.

## Per-class comparison with Table 8

| Species | Paper baseline mAP@50 | First-run mean mAP@50 | Difference (pp) |
| --- | ---: | ---: | ---: |
| rice leaf roller | 47.2% | 43.51% | -3.69 |
| pink rice borer | 75.2% | 72.19% | -3.01 |
| asiatic rice borer | 25.8% | 15.51% | -10.29 |
| yellow rice borer | 53.0% | 55.31% | +2.31 |
| rice leaf caterpillar | 43.5% | 40.30% | -3.20 |
| small brown plant hopper | 65.6% | 55.59% | -10.01 |
| white backed plant hopper | 56.0% | 28.83% | -27.17 |
| brown plant hopper | 46.2% | 49.64% | +3.44 |
| rice leaf hopper | 69.5% | 65.71% | -3.79 |
| rice gall midge | 74.7% | 73.34% | -1.36 |
| paddy stem maggot | 61.1% | 55.87% | -5.23 |
| rice stem fly | 81.9% | 81.89% | -0.01 |
| grain spreader thrips | 65.3% | 63.41% | -1.89 |
| rice weevil | 65.5% | 60.69% | -4.81 |
| rice shell pest | 76.8% | 76.81% | +0.01 |
| melanitis leda | 71.1% | 73.94% | +2.84 |
