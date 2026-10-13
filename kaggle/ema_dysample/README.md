# EMA + DySample single-run experiment

The architecture in `configs/paddy-yolo-ema-dysample.yaml` follows the supplied
figure: EMA follows the P3/P4/P5 RGG backbone stages; DySample replaces both
top-down neck upsampling layers; the PAN paths and three Detect scales remain.
The diagram's unscaled channel labels are reduced by the baseline's nano width
factor (0.25) and depth factor (0.33). EMA here means Efficient Multi-Scale
Attention; Ultralytics also maintains an independent moving average of weights.

Run configuration: `run_config.json`. Account `dmrashidpferrer`, **300 epochs**,
total batch **32 across GPUs 0 and 1 (16 per GPU)**, image size **640**, AdamW
with `configs/hyp_ema_dysample.yaml` (`lr0=0.001`, `warmup_bias_lr=0.0`), DIMIoU H,
seed 42, patience 300 and an additional checkpoint every 25 epochs.

`yolov8n.pt` supplies COCO pretrained weights. The transfer maps each source
layer to the corresponding enhanced-model layer, including the shifted backbone,
SPPF, neck and Detect. Shape-compatible tensors transfer; new EMA/DySample
parameters and incompatible COCO class-output tensors retain initialization.
The transfer counts and mapping are saved to `weight_transfer.json`.

Dataset: [R2000-PestNet](https://www.kaggle.com/datasets/dmrashidpferrer/r2000-pestnet),
using its supplied 1,214 training, 304 validation and 506 test images. Class IDs
and generic names are preserved. `fold0` labels this single run; it does not
regenerate five-fold splits. The dataset audit rejects cross-split duplicate
images and corrupt annotations. Test data is evaluated after training selects
`best.pt` using validation.

Build from the repository root:

```powershell
.\.venv\Scripts\python.exe kaggle/ema_dysample/build_notebook.py --username dmrashidpferrer
```

The notebook embeds its source and retains Kaggle's CUDA-enabled PyTorch.
It disables unused external experiment trackers before tests and training.
Kaggle's Ray version is incompatible with Ultralytics 8.3.18's old callback;
its W&B callback also rejects filesystem paths as tracker project names.
Ultralytics metrics also use NumPy's supported trapezoidal integrator when the
legacy `trapz` function is absent. Setup, preflight tests and training are logged
to downloadable `environment_setup.log`, `preflight_tests.log` and
`ema_dysample_output/training.log`.
Local/Kaggle API credentials must be provided with `KAGGLE_API_TOKEN`; no token
is stored in notebook source or metadata. Push starts the authorized private run:

```powershell
.\.venv\Scripts\kaggle.exe kernels push -p kaggle/ema_dysample --accelerator NvidiaTeslaT4
.\.venv\Scripts\kaggle.exe kernels status dmrashidpferrer/paddy-yolo-ema-dysample-rgg-fold0
.\.venv\Scripts\kaggle.exe kernels output dmrashidpferrer/paddy-yolo-ema-dysample-rgg-fold0 -p results/ema_dysample/rgg_fold0
```

Outputs: `ema_dysample_output/train/ema_dysample_rgg_fold0/weights/best.pt`, `last.pt`,
periodic checkpoints, `results.csv`, validation/test evaluation plots,
`ema_dysample_output/metrics.json` and `weight_transfer.json`.
Kaggle runtime or quota limits can end a session before 300 epochs; keep `last.pt`
to resume from a downloaded checkpoint. The runner accepts `--resume --weights PATH`.

Algorithm references: [EMA implementation](https://github.com/YOLOonMe/EMA-attention-module),
[DySample reference implementation](https://github.com/tiny-smart/dysample).
DySample is adapted under its included MIT license. The LP variant uses four
sampling groups, 2x scale and no dynamic scope branch. EMA uses 32 channel groups.

## Verified fixes and restart

On 2026-10-13, the private Kaggle diagnostic passed all **100 tests**, trained
one full R2000 epoch on **both T4 GPUs** with total batch **32** (16 per GPU),
saved and reloaded `best.pt`, and completed validation/test evaluation. It used
Kaggle's PyTorch 2.11.0+cu128 with Ultralytics 8.3.18.
See `verification.json` for the check results and `launch-original-fold0-v2.json`
for the original 300-epoch job (notebook version 2). The failed version 1 record is retained in
`launch-v1.json`.

The fixes disable unrelated tracker callbacks, retain full subprocess logs,
adapt the legacy AP integrator for newer NumPy, expose the source package to DDP
workers, and unwrap parallel models before installing the custom validation loss.

## Completed results

Notebook version 2 completed all 300 epochs. Its downloaded artifacts are in
[`results/ema_dysample`](../../results/ema_dysample/summary.md), with the remote
`ema_dysample_output/` prefix removed. The best checkpoint's validation mAP50 is
64.07% and mAP50-95 is 45.36%; test mAP50 is 59.03% and mAP50-95 is 41.75%.
The folder includes checkpoints, training history, evaluation plots, logs,
configuration records, and a download manifest with SHA-256 checksums.

## Corrected RGG single-fold rerun

The current `launch.json` records the new private notebook
`dmrashidpferrer/paddy-yolo-ema-dysample-rgg-fold0`. RGG now registers every
parameter and buffer once so Ultralytics weight averaging updates each tensor
once. Local checks passed 111 tests, including RGG/EMA/DySample gradients and
legacy checkpoint loading. This run also tests lower AdamW learning rates;
the model still initializes from `yolov8n.pt` and trains 300 epochs on the same
single supplied split. Retrieve its results under `results/ema_dysample/rgg_fold0`
to preserve the original results. Full changes are recorded in
[`RGG_EMA_DySample_changes.txt`](../../RGG_EMA_DySample_changes.txt).

