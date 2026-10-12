# Baseline replication run

The authorized experiment was submitted as a private Kaggle GPU notebook:
[Paddy-YOLO baseline fivefold R2000](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-baseline-fivefold-r2000-20261011).

It runs standard YOLOv8-N with CIoU from random initialization, 300 epochs per
fold, batch size 32, 640x640 inputs, AdamW and `configs/hyp_paper.yaml`. All five
folds run sequentially on CUDA device 0. The same prepared fold manifests can
be used for the proposed model to keep the comparison consistent. Replication
assumptions are described in [the audit](replication-audit.md).

The dataset is mounted from `dmrashidpferrer/r2000-pestnet`. The pipeline verifies
2,024 images and 2,281 boxes, converts the generic class names to the paper's
16 species, and regenerates grouped fivefold splits. Each iteration uses three
training folds, one validation fold and one held-out test fold. Validation selects
`best.pt`; test metrics do not select checkpoints.

Outputs under `baseline_results/` include:

- `run_manifest.json`: settings, environment, dataset counts, current/completed folds and status.
- `fold0_training.log` through `fold4_training.log`.
- `train/baseline_yolov8n_fold*/weights/best.pt` and `last.pt`.
- `fold0_test_metrics.json` through `fold4_test_metrics.json`, including per-class results.
- `individual_folds.csv` and `fivefold_summary.json`, with means and sample standard deviations.
- Training and test plots, results CSVs and fold manifests.

The API token is used only in the local launching/checking processes' environment.
It is not embedded in scripts, source archives, notebooks, metadata or result files.
The upload archive uses an explicit project-file allowlist.

The October 12 corrective rerun uses FP32 validation while retaining AMP training,
sets AdamW bias warmup to zero, and rejects incomplete/non-finite epoch histories.
See [the paper comparison and numerical findings](baseline-results-2026-10-12.md).

Build an equivalent private notebook with:

```powershell
.\.venv\Scripts\python.exe build_baseline_crossval_kernel.py --owner YOUR_KAGGLE_USERNAME
```

With `KAGGLE_API_TOKEN` supplied in the process environment, check the submitted job:

```powershell
.\.venv\Scripts\kaggle.exe kernels status dmrashidferrer/paddy-yolo-baseline-fivefold-r2000-20261011
.\.venv\Scripts\kaggle.exe kernels logs dmrashidferrer/paddy-yolo-baseline-fivefold-r2000-20261011
```

When execution finishes, download the saved outputs:

```powershell
.\.venv\Scripts\kaggle.exe kernels output dmrashidferrer/paddy-yolo-baseline-fivefold-r2000-20261011 -p data/baseline_crossval_results
```

A submitted or running job does not establish published baseline accuracy.
Report the actual fivefold test results after all folds finish. Kaggle time or
quota limits can interrupt execution; completed folds and checkpoints are
written individually as the run progresses.
