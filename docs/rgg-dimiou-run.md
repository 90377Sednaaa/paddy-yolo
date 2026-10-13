# Final training target: Paddy-YOLO with four RGG stages and DIMIoU

The user's final correction selects the final proposed model in Tables 8 and 13
of the supplied paper: **four sequential RGG backbone stages + DIMIoU Strategy H**.
The earlier CIoU baseline and fixed IMPDIoU experiment are separate configurations.

`configs/paddy-yolo.yaml` has RGG modules at layers 2, 4, 6 and 8. Each takes
the previous backbone layer as input, with downsampling Conv layers between
stages as shown in Figure 5. Each internal RGGConv follows RepConv -> GhostConv
-> GhostConv sequentially, with the skip concatenations in Figure 6(b).
The neck and three detection heads retain the YOLOv8-N layout.

The bounding-box criterion is `PaddyDetectionLoss` / `PaddyBboxLoss` with
`strategy="H"` and **no fixed ratio**. Equation (19) computes the dynamic ratio
`0.5*cos((1-IoU)*pi)+1`, giving 0.5, 1.0 and 1.5 at IoUs 0, 0.5 and 1.
BCE classification and DFL remain active. The loss normalization assumption
is disclosed in [the replication audit](replication-audit.md).

| Paper's final Paddy-YOLO metric on R2000 | Value |
| --- | ---: |
| Precision | 73.1% |
| Recall | 63.0% |
| F1 | 67.7% |
| mAP@50 | 70.2% |
| mAP@50:95 | 51.5% |

The experiment uses 300 epochs per fold, batch 32, 640-pixel inputs, AdamW,
random initialization and fivefold 3/1/1 splits. It retains the numerical repairs:
FP32 validation, AdamW bias warmup 0.0 and strict finite-loss/history checks.

The saved-checkpoint test reloads a trained model and verifies both the RGG
modules and the DIMIoU H criterion without a fixed ratio. The runtime preflight
checks the exact RGG indices, their sequential inputs and the loss configuration.
These checks establish which model is trained; accuracy requires the completed
fivefold evaluation and comparison with the paper.

The intended private notebook is
[RGG + DIMIoU fivefold](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-rgg-dimiou-fivefold-20261012).
Build it using:

```powershell
.\.venv\Scripts\python.exe build_baseline_crossval_kernel.py --owner dmrashidferrer --slug paddy-yolo-rgg-dimiou-fivefold-20261012 --experiment rgg-dimiou
```

Outputs go to `rgg_dimiou_results/`, including explicit architecture/loss metadata,
all five checkpoints, per-fold test metrics and the mean/sample standard deviation.

Version 1 completed on October 12, 2026. The outputs are downloaded under
`data/rgg_dimiou_results_download/rgg_dimiou_results/` (Git-ignored), including
`fivefold_summary.json`, `individual_folds.csv`, five per-fold test-metric files,
five training histories and best/last checkpoints for every fold.

## Held-out test results

These are means and sample standard deviations across the five **test** folds;
the paper values are the reported final Paddy-YOLO values from Tables 8/13.

| Metric | This run, mean ± sample SD | Paper | Difference |
| --- | ---: | ---: | ---: |
| Precision | 51.81% ± 2.29 pp | 73.1% | −21.29 pp |
| Recall | 53.34% ± 1.60 pp | 63.0% | −9.66 pp |
| F1 | 52.52% ± 1.06 pp | 67.7% | −15.18 pp |
| mAP@50 | 54.39% ± 1.44 pp | 70.2% | −15.81 pp |
| mAP@50:95 | 38.66% ± 1.40 pp | 51.5% | −12.84 pp |

| Fold | mAP@50 | mAP@50:95 |
| ---: | ---: | ---: |
| 0 | 54.55% | 38.15% |
| 1 | 53.06% | 37.32% |
| 2 | 54.44% | 38.48% |
| 3 | 56.68% | 41.04% |
| 4 | 53.24% | 38.32% |

The manifest reports all five folds complete, 2,024 images, 2,281 boxes,
four sequential RGG stages, DIMIoU Strategy H, and random initialization.
Each per-fold test file covers 16 classes. All five `results.csv` histories have
exactly 300 sequential epochs with finite recorded values. The downloaded
best and last checkpoints for every fold passed ZIP integrity checks. Summary
means and sample standard deviations independently recomputed from
`individual_folds.csv` match `fivefold_summary.json`.

This run falls short of the paper's reported accuracy. The downloaded files
confirm the measurements and run settings; they do not identify the cause of
the accuracy gap. The paper's exact fold membership and several architecture
and loss details remain uncertain, as described in the replication audit.
