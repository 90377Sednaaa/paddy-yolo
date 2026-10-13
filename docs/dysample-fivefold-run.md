# Plain Paddy-YOLO-DySample five-fold run

Submitted on 2026-10-13 (Asia/Manila) as private [Kaggle notebook version 1](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-r2000-fivefold). Kaggle reported COMPLETE when retrieved on 2026-10-13. All five folds completed 300 epochs and held-out test evaluation; metrics and all 40 checkpoints were downloaded and verified. See the [results and baseline comparison](dysample-fivefold-results.md). The notebook was generated in `kaggle_dysample_fivefold_kernel/` by `python build_dysample_crossval_kernel.py --owner dmrashidferrer`.

## Experiment

All five folds, including fold 0, train afresh from compatible `yolov8n.pt` weights. No fold loads the previous fold's trained weights or the earlier single-fold result. The architecture is `configs/Paddy-YOLO-DySample.yaml`, with `scale=2`, `style='lp'`, `groups=4`, `dyscope=False` at both neck upsamplers. RGG, DIMIoU Strategy H, the three detection scales and dataset preparation use their existing implementations.

Each fold trains for 300 epochs with `configs/hyp_paper.yaml`, AdamW, global batch 32, image size 640, seed 42 and checkpoint period 50. Two visible GPUs use one DDP job with batch 16 per worker; the existing single-GPU fallback uses batch 32. Folds run sequentially to share the same accelerator allocation. At the earlier two-T4 speed of about 70 minutes per fold, estimated total execution is roughly six hours, including setup and test evaluation. This is an estimate, not a guaranteed completion time.

The existing deterministic, duplicate-aware split uses three partitions for training, one for validation and one for testing. After a fold finishes, its epoch CSV must contain exactly epochs 1 through 300 with finite values. Its best checkpoint, selected during validation, is then evaluated on that fold's held-out **test** split using FP32. These test scores are distinct from the earlier single-fold validation scores.

## Outputs

The output root is `/kaggle/working/dysample_fivefold/`:

- `proposed/fold0/` through `proposed/fold4/`: independent training CSVs, plots and checkpoints. `last.pt` is saved each epoch; periodic files use zero-based names `epoch0.pt`, `epoch50.pt`, etc.
- `test/foldN/`: held-out test plots and prediction previews.
- `foldN_training.log`: streamed training output, also forwarded to the notebook console.
- `foldN_test_metrics.json`: exact overall and per-class test values, counts and training-history verification.
- `individual_folds.csv`: one overall test row per completed fold.
- `per_class_and_overall.csv`: 17 rows per fold (overall plus 16 classes), totaling 85 when complete.
- `fivefold_summary.json` and `fivefold_summary.csv`: unweighted overall means and sample standard deviations, written only after all five unique folds finish.
- `per_class_summary.csv`: per-class means, sample standard deviations and the number of folds in which the class was evaluated.
- `run_manifest.json`: status, current/completed folds, GPUs, initialization, architecture/hyperparameter hashes and split-membership hashes.

Metric columns are native fractions from 0 to 1. Classes absent from a particular test split have unavailable metrics, represented by JSON nulls or empty CSV cells rather than invented zeros. The supported-fold count is reported in the per-class summary.

## Recovery in a later session

Restore the original **output tree** to `/kaggle/working/dysample_fivefold/` before continuing. It must include `run_manifest.json`, earlier folds' reports, and the interrupted fold's `results.csv`, unstripped checkpoint and associated `best.pt`. Mounting an isolated checkpoint is insufficient.

Set `START_FOLD` to the current fold in the notebook. If training was interrupted, set `RESUME_FROM` to the restored checkpoint within that fold's weights directory, for example `/kaggle/working/dysample_fivefold/proposed/fold2/weights/last.pt`. Use an unstripped checkpoint with optimizer/scaler state. Checkpoint settings, full architecture, restored CSV history and best-checkpoint fitness must match before training resumes. The CSV is restored to the checkpoint's saved epoch snapshot to prevent duplicate history when resuming from an older checkpoint; later folds still initialize from `yolov8n.pt`.

If training already completed all 300 epochs but test evaluation failed, use that fold as `START_FOLD` and leave `RESUME_FROM = None`. The runner verifies the complete history and retries test evaluation without repeating training.

## Local verification

Before submission, the full local suite passed 159 tests. New tests cover fresh initialization on all five folds, correct class-ID mapping, finite metrics, complete-fold summary requirements and sample standard deviation, partial-result preservation, restoring both GPUs after standalone test evaluation, continuing completed folds, rejecting incompatible resume settings/architecture/history, restoring checkpoint history and retrying test evaluation after completed training. A real FP32 validator smoke check on a synthetic held-out split also passed. Full five-fold execution has now completed on two T4 GPUs, and the downloaded histories, metric summaries and checkpoints passed retrieval verification.
