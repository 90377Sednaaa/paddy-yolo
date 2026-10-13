# Paddy-YOLO-DySample: R2000 fold 0 results

[Kaggle run](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-r2000-fold0), version 4, completed on 2026-10-12. All 300 epochs finished using two T4 GPUs, global batch 32, image size 640, AdamW, DIMIoU Strategy H and compatible pretrained tensors from `yolov8n.pt`. Training used `configs/hyp_paper.yaml`. The recorded training duration was 1.172 hours (about 70.3 minutes); total notebook execution was about 72.5 minutes.

The saved `results.csv` contains epochs 1 through 300, with all numeric entries finite. `best.pt` contains the same metrics as epoch 287, selected by mAP@50-95. The final epoch has slightly lower mAP than the selected checkpoint.

| Saved validation metric | Best checkpoint, epoch 287 | Last checkpoint, epoch 300 |
| --- | ---: | ---: |
| Precision | 70.619% | 69.029% |
| Recall | 59.542% | 59.662% |
| mAP@50 | 68.115% | 66.616% |
| mAP@50-95 | 51.066% | 50.201% |

Validation covered 405 images and 456 labeled instances. These are fold 0 validation results. The held-out test split was not evaluated by this notebook, and no baseline comparison or five-fold average is established by this run.

Final revalidation of the stripped best checkpoint printed mAP@50 0.681 and mAP@50-95 0.511, consistent with the saved checkpoint metrics after rounding. The weakest class by mAP@50 was asiatic rice borer (33.1%, recall 29.2%, 24 instances). The strongest was melanitis leda (89.9% mAP@50, 22 instances), followed by rice stem fly (87.7%, 73 instances).

Downloaded artifacts are under `data/kaggle_dysample_full_results/dysample_training/proposed/fold0/`: `results.csv`, `args.yaml`, training curves, precision-recall/F1 curves, confusion matrices, and `weights/best.pt`, `weights/last.pt`, `weights/epoch250.pt`. The complete execution log is `data/kaggle_dysample_full_complete_v4.log`.

All three downloaded checkpoints loaded successfully and contain the two DySample neck modules. `best.pt` and `last.pt` are completed-run inference checkpoints: Ultralytics stripped their optimizer/scaler state and set epoch to -1. `epoch250.pt` retains optimizer and scaler state with zero-based epoch 250 (251 completed epochs), so it supports resuming. The earlier DDP validation-wrapper and missing-rank-metrics failures did not recur during this completed run.
