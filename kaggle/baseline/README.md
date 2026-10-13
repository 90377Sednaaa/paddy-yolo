# Kaggle baseline

This runs the current `Paddy-Yolo-Baseline` implementation: RGG backbone,
GhostConv/RepConv, and DIMIoU Strategy H. For a standard YOLOv8-N + CIoU
comparison, set `MODEL_KIND = "yolov8n"` in the notebook.

## Run on Kaggle

1. Import `baseline.ipynb` into a new Kaggle notebook.
2. Add [R2000-PestNet](https://www.kaggle.com/datasets/dmrashidpferrer/r2000-pestnet)
   as an input. Enable **GPU** and **Internet** in notebook settings.
3. Run all cells. The default is one epoch at 640 pixels with batch size 16.
   This validates the pipeline; it does not establish model accuracy.
4. For longer training, start a fresh session and change `EPOCHS` to `300`.
   The paper batch size is 32; increase `BATCH` if GPU memory permits.
   Kaggle session limits may require a separate checkpoint/resume workflow.

The notebook embeds the project's source, so a Git clone or extra code dataset
is unnecessary. It installs `ultralytics==8.3.18` and retains Kaggle's CUDA
PyTorch installation. It runs the full verification and test suite before training.
COCO transfer downloads `yolov8n.pt`; set `SCRATCH = True` to train from scratch.

The dataset already contains 1,214 training, 304 validation and 506 test images
(2,024 total, 2,281 boxes). Its class names are `classID_00` through `classID_15`;
no verified mapping to semantic pest names is supplied. We preserve those IDs,
names and split assignments. Preparation checks every image, annotation and image
hash, rejecting cross-split duplicates. It copies labels and links images into
a writable working directory for Ultralytics caching.

Do not use the old `build_5fold_baseline_kernel.py` for this dataset: it references
a different dataset and regenerates folds. This run uses the supplied fixed split.
Shuffling duplicate images into new folds can introduce leakage.

Outputs are under `/kaggle/working/baseline_output`: training checkpoints,
`results.csv`, evaluation plots and `metrics.json` with validation and test metrics.
`baseline_data/dataset_audit.json` records split sizes and class counts.
`best.pt` is selected using validation, then evaluated on the held-out test split.
Use a new output directory or fresh session for each run; existing results are protected.

## Rebuild after source changes

From the repository root:

```powershell
.\.venv\Scripts\python.exe kaggle/baseline/build_notebook.py
```

Only the explicit source allowlist is embedded. Credentials, environments, data,
weights and existing outputs are excluded.

## Optional Kaggle API workflow

An attached Kaggle dataset requires no API key in the notebook. For local API
commands, use `KAGGLE_API_TOKEN`, as described in the
[official authentication documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#authentication).
Enter the token through a hidden prompt instead of saving it in source or shell history:

```powershell
$credential = Read-Host 'Kaggle API token' -AsSecureString
$env:KAGGLE_API_TOKEN = [System.Net.NetworkCredential]::new('', $credential).Password
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\kaggle.exe datasets files dmrashidpferrer/r2000-pestnet
```

To upload and start a private notebook, first rebuild metadata with your actual
Kaggle username, then push when you are ready to consume GPU quota:

```powershell
.\.venv\Scripts\python.exe kaggle/baseline/build_notebook.py --username YOUR_KAGGLE_USERNAME
.\.venv\Scripts\kaggle.exe kernels push -p kaggle/baseline
.\.venv\Scripts\kaggle.exe kernels status YOUR_KAGGLE_USERNAME/paddy-yolo-r2000-pestnet-baseline
.\.venv\Scripts\kaggle.exe kernels output YOUR_KAGGLE_USERNAME/paddy-yolo-r2000-pestnet-baseline -p kaggle/baseline/output
Remove-Item Env:KAGGLE_API_TOKEN
```

`kernel-metadata.json` is configured for a private GPU notebook with the correct
dataset attached. Its username placeholder must be replaced before an API push.
No notebook has been uploaded or started by preparing these files.

## Local setup verification (2026-10-13)

The dependency check and all six implementation verification stages passed on
Python 3.13.0, PyTorch 2.7.1+cpu, Torchvision 0.22.1+cpu and Ultralytics 8.3.18.
All **84 tests passed** when run independently against the source unpacked from
the generated notebook. This includes the dataset safeguards and both training modes.

The full public dataset was audited locally. See [dataset_audit.json](dataset_audit.json)
for class counts, split sizes, 12 duplicate groups within splits, and zero duplicates
crossing splits. An eight-image local CPU smoke run completed one epoch with COCO
weight transfer, saved checkpoints, and generated validation/test metrics. Its zero
mAP is expected for this tiny pipeline check and is not an accuracy benchmark.

Fixed setup issues: two integration tests depended on a COCO download; standard
`train.py --baseline` passed unsupported custom loss arguments to Ultralytics.
Both training modes now pass local checks. Kaggle GPU execution is still untested;
this delivery only builds the notebook and metadata.
