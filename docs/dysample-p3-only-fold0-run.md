# P4-to-P3-only DySample: R2000 fold 0

Submitted on 2026-10-13 (Asia/Manila) as version 1 of the private [dmrashidferrer notebook](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-p3-only-r2000-fold0), requesting `NvidiaTeslaT4` acceleration. Kaggle confirmed submission and reported RUNNING. The pulled notebook's source matches the validated local notebook. This experiment schedules only fold 0. Full training and test results are pending.

The separate configuration is `configs/Paddy-YOLO-DySample-P3.yaml`. Relative to the original Paddy-YOLO, only model layer 13 changes:

`C2f 12 → DySample 13 → Concat with backbone P3 at 14 → C2f 15`

[Architecture diagram (SVG)](diagrams/paddy-yolo-dysample-p3-fold0.svg) · [PNG](diagrams/paddy-yolo-dysample-p3-fold0.png). Feature dimensions were checked with a 640px forward pass; the orange block marks the single DySample module.

Layer 10 retains nearest-neighbor P5-to-P4 upsampling. Layer 13 uses the original DySample implementation with 128 input channels, `scale=2`, `style='lp'`, `groups=4`, `dyscope=False`; it converts a 128×40×40 feature map to 128×80×80 at 640px input. RGG, DIMIoU Strategy H, concatenation routes, three detection scales, dataset preparation and baseline/two-stage DySample configurations remain unchanged. No EMA attention or BiFPN is included in this model. Total parameters are 3,379,296, adding 4,128 to the baseline.

Fold 0 starts afresh from compatible `yolov8n.pt` weights. Initialization is seeded before model construction, using the previously corrected training path. It keeps `configs/hyp_paper.yaml`, 300 epochs, AdamW, global batch 32, image size 640, seed 42, workers 2 and checkpoint period 50. Two visible T4 GPUs use DDP with batch 16 per worker; a single GPU uses batch 32. The generated notebook streams training output and runs one training command, followed by held-out test evaluation.

The same deterministic duplicate-aware R2000 split uses three partitions for training, one validation partition and one held-out test partition. Validation selects `best.pt`; after verifying exactly 300 finite epoch rows, the runner evaluates that checkpoint on the test split using FP32. The result is a one-fold test result. It should be compared with previous **fold 0 test** metrics, rather than earlier single-fold validation scores or five-fold means.

Comparison limitations still apply: the earlier completed runs constructed custom weights before setting the trainer seed, and the pretrained RGG + DIMIoU run used one GPU. Consequently, the previous outputs are useful reference results but cannot fully isolate placement as the sole cause of any difference. A matched repeat of the baseline/two-stage model would be a separate experiment; this notebook trains only the requested P3-only fold.

The output root is `/kaggle/working/dysample_p3_training/`:

- `p3_only/fold0/`: training CSV, arguments, plots and checkpoints.
- `p3_only/fold0/weights/`: `best.pt`, `last.pt` and periodic `epoch0.pt`, `epoch50.pt`, …, `epoch250.pt`.
- `test/fold0/`: held-out test curves, confusion matrices and prediction previews.
- `fold0_training.log`: streamed training output.
- `fold0_test_metrics.json`: exact overall/per-class test metrics, counts and history verification.
- `individual_folds.csv`: one overall row.
- `per_class_and_overall.csv`: 17 rows, overall plus all 16 classes. Metrics use native fractions from 0 to 1; unavailable class metrics are empty rather than invented zeros.
- `run_manifest.json`: status, selected fold, initialization, GPU allocation, DySample placement, architecture/hyperparameter hashes and split-membership hashes.

For recovery in another session, restore the output tree to the same path, including manifest, training CSV, associated best checkpoint and an unstripped resume checkpoint. Set `RESUME_FROM` in the notebook to that checkpoint inside `p3_only/fold0/weights/`. The runner checks the original model, fold, hyperparameters, split hashes, checkpoint state and saved history before resuming. If training finished all 300 epochs but evaluation failed, leave `RESUME_FROM = None`; the runner verifies the complete history and retries test evaluation without another training run.

Implementation files are `configs/Paddy-YOLO-DySample-P3.yaml`, `run_dysample_p3_experiment.py`, the new variant in `run_dysample_experiment.py`, P3-only checkpoint validation in `run_dysample_crossval.py`, and packaging/selection in `build_dysample_kaggle_kernel.py` and `build_dysample_full_kernel.py`. Build with `python build_dysample_full_kernel.py --owner dmrashidferrer --fold 0 --variant p3_only`; outputs are in `kaggle_dysample_p3_kernel/`.

The final full suite passed 172 tests, including 32 focused checks. Full 640px forward propagation preserves P3/P4/P5 inputs `(1,64,80,80)`, `(1,128,40,40)`, `(1,256,20,20)` and prediction shape `(1,20,8400)`. Backward propagation reaches the DySample offset weights with finite nonzero gradients. An actual one-epoch synthetic CPU training run completed with finite losses, checkpoint reload preserving DIMIoU H, and successful FP32 held-out test evaluation. Its scores are execution checks, not R2000 performance estimates. Smoke evidence is in `data/dysample_p3_validation/verification.json` and `data/dysample_p3_smoke.log`. Independent code review found no critical or important issues. Packaging verification confirms all embedded sources match the workspace, compile, and exclude credentials; evidence is in `data/dysample_p3_launch_verification.json`.
