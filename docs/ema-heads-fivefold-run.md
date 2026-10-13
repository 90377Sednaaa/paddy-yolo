# Paddy-YOLO with EMA before P3/P4/P5

Submitted on 2026-10-13 (Asia/Manila) as version 1 of the private [dmrashidpferrer Kaggle notebook](https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-ema-heads-r2000-fivefold), requesting `NvidiaTeslaT4` acceleration. Kaggle confirmed successful submission and reported RUNNING afterward. Full training results are pending.

The requested experiment uses [Paddy-YOLO-EMA-Heads.yaml](../configs/Paddy-YOLO-EMA-Heads.yaml). The original RGG backbone, DIMIoU Strategy H, baseline configurations and DySample experiments remain available independently. Both fusion-neck upsamplers use nearest-neighbor interpolation.

## Attention placement

The existing project's Efficient Multi-Scale Attention module is applied in three detection branches. Its default grouping factor is eight, and its existing residual output `x + attention(x)` is retained.

| Detection scale | Source feature | EMA layer | Channels at 640px | Spatial size |
| --- | ---: | ---: | ---: | --- |
| P3 / stride 8 | 15 | 22 | 64 | 80 × 80 |
| P4 / stride 16 | 18 | 23 | 128 | 40 × 40 |
| P5 / stride 32 | 21 | 24 | 256 | 20 × 20 |

Detect moves from layer 22 to layer 25 and consumes `[22, 23, 24]`. Layers 0 through 21 match the baseline architecture exactly. EMA outputs feed detection only, preserving the original features used in subsequent fusion stages. The older `configs/paddy-yolo-ema.yaml` instead places attention within the bottom-up fusion path; it is preserved as a separate architecture.

The original model has 3,375,168 trainable parameters and the proposed model has 3,388,720, an increase of 13,552. No inference speed or FLOP claim is made here; these values are direct model parameter counts.

## Initialization and training

Every fold, including fold 0, starts afresh from compatible `yolov8n.pt` weights. No fold imports a trained previous fold or an earlier EMA run. The shifted detection layer receives 49 compatible pretrained tensors by local tensor name and shape, in addition to 234 compatible tensors transferred using their unchanged model indices. COCO classification tensors with incompatible shapes are initialized for the 16 pest classes.

Seeding now runs before construction of the pretrained custom architecture, so the requested seed controls initial EMA, unmatched RGG and classification weights as well as subsequent training. Earlier experiment code constructed these weights before the trainer seeded its RNG; keep that initialization difference in mind when comparing previous completed runs.

All five folds use the existing deterministic, duplicate-aware R2000 split: three partitions train, one validates and one tests. Training keeps `configs/hyp_paper.yaml`, 300 epochs per fold, AdamW, total batch 32, 640px images, seed 42, two workers and checkpoint period 50. Two visible GPUs use DDP with batch 16 per worker; the existing one-GPU fallback uses batch 32. Folds execute sequentially.

Training output is streamed with Python buffering disabled. Each best checkpoint is selected using validation, then evaluated separately on its held-out test split using FP32. Overall and per-class exact metrics, counts, five-fold means and sample standard deviations are exported using the same reporting pipeline as the plain DySample five-fold run.

## Output and recovery

The output root is `/kaggle/working/ema_heads_fivefold/`:

- `ema_heads/foldN/`: training arguments, epoch histories, plots, and `weights/{best,last}.pt` plus periodic checkpoints.
- `test/foldN/`: held-out test curves, confusion matrices and prediction previews.
- `foldN_training.log` and `foldN_test_metrics.json`: training log and exact test metrics.
- `individual_folds.csv`, `per_class_and_overall.csv`, `fivefold_summary.{json,csv}` and `per_class_summary.csv`: result exports.
- `run_manifest.json`: status, completed folds, attention placement, initialization, GPU list, architecture/hyperparameter SHA-256 hashes and split-membership hashes.

To continue in another session, restore this output tree to the same working path. It must contain the manifest, preceding completed folds' reports, and the interrupted fold's epoch CSV, unstripped checkpoint and associated best checkpoint. Set `START_FOLD` and `RESUME_FROM` in the notebook. The runner verifies architecture, hyperparameters, fold identity and checkpoint-history consistency before resuming; later folds initialize from `yolov8n.pt`. If all 300 training epochs completed but test evaluation failed, set that fold as `START_FOLD` with `RESUME_FROM = None` to retry evaluation.

## Modified files and local verification

- New architecture: `configs/Paddy-YOLO-EMA-Heads.yaml`.
- New notebook builder: `build_ema_heads_crossval_kernel.py`; generated notebook directory `kaggle_ema_heads_fivefold_kernel/`.
- `run_dysample_experiment.py`: registers the independent `ema_heads` experiment.
- `run_dysample_crossval.py`: reuses the tested five-fold reporting/recovery loop for the EMA variant; the default plain DySample behavior is retained.
- `build_dysample_kaggle_kernel.py`: packages the new configuration alongside the reusable training sources.
- `paddy_yolo/modules/register.py`: corrects the inferred output-channel count for channel-preserving modules referencing earlier features, as required for independent EMA branches.
- `train.py`: transfers compatible shifted detection weights for this architecture and seeds custom-model initialization before construction.
- `tests/test_ema_heads_experiment.py`: tests feature routes, 640px detection shapes, backward propagation, pretrained transfer, deterministic initialization, fresh five-fold execution, packaging and checkpoint recovery.

The final local suite passed 166 tests. Local forward propagation verified `(1, 64, 80, 80)`, `(1, 128, 40, 40)` and `(1, 256, 20, 20)` detection inputs and `(1, 20, 8400)` predictions. Backward propagation produced finite gradients through all three EMA modules. A one-epoch synthetic CPU training run completed with finite losses, a reloadable DIMIoU H checkpoint and successful FP32 held-out test evaluation. That smoke check tests execution and compatibility; its synthetic metrics do not predict R2000 accuracy. Evidence is stored in `data/ema_heads_validation/verification.json` and `data/ema_heads_smoke_final.log`. All 29 packaged source files match the final workspace files; Python sources and notebook cells compile, and credentials are excluded. Packaging evidence is in `data/ema_heads_launch_verification.json`.
