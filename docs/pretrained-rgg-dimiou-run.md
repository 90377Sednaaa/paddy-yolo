# Pretrained Paddy-YOLO Kaggle run

The requested training notebook was submitted as version 1 to the account
authenticated by the supplied token, `dmrashidpferrer`:
[Paddy-YOLO pretrained RGG + DIMIoU](https://www.kaggle.com/code/dmrashidpferrer/paddy-yolo-pretrained-rgg-dimiou-20261012).
The notebook completed all five folds. Its results and checkpoints were retrieved
on October 13, 2026 (Asia/Manila); see [the completed results report](pretrained-rgg-dimiou-results.md).
The five-fold held-out test means are mAP@50 **64.31%** and mAP@50-95 **46.34%**.

This private GPU notebook runs standard `configs/paddy-yolo.yaml`, with four
RGG backbone stages and DIMIoU Strategy H, on R2000. Each of the five folds
starts independently from COCO-pretrained `yolov8n.pt`, using 300 epochs,
batch 32, 640-pixel inputs, AdamW and seed 42. Compatible tensors transfer by
name and shape; the checkpoint does not provide pretrained custom RGG blocks.
Pretrained initialization is a separate experiment from the random-initialization
replication described in [the original run notes](rgg-dimiou-run.md).

The runtime preflight checks the four RGG stages and DIMIoU criterion, loads
the pretrained checkpoint, verifies exact first-convolution weight transfer,
and records the checkpoint SHA-256 and verification status in the run manifest.
Each fold selects `best.pt` on validation data and reports held-out test metrics.
Results are saved under `/kaggle/working/rgg_dimiou_results/`.

Rebuild with:

```powershell
python build_baseline_crossval_kernel.py --owner dmrashidpferrer --slug paddy-yolo-pretrained-rgg-dimiou-20261012 --experiment rgg-dimiou --weights yolov8n.pt
```

The submitted package is preserved in `kaggle_pretrained_kernel/` (Git-ignored).
The token was supplied only to local command processes and is excluded from
the source archive, notebook, metadata and documentation.
