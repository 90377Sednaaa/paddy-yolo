# Paddy-YOLO-DySample+ experiment

The approved follow-up experiment enables the official DySample dynamic scope branch at both neck upsampling layers, model layers 10 and 13. Its separate configuration is `configs/Paddy-YOLO-DySample-Plus.yaml`, and the runner selects it with `--variant proposed_plus`.

Both layers retain `scale=2`, `style='lp'` and `groups=4`; only `dyscope` changes from `False` to `True`. The original DySample implementation and license remain the same. The RGG backbone, DIMIoU criterion, detection heads and dataset preparation use the existing implementations. The baseline and plain DySample configurations remain selectable independently.

| Local verification | Result |
| --- | --- |
| Trainable parameters | 3,399,792 (+12,288 over plain DySample) |
| Compatible tensors transferred from `yolov8n.pt` | 283, checked for exact equality |
| P3/P4/P5 inputs at 640 pixels | `[1,64,80,80]`, `[1,128,40,40]`, `[1,256,20,20]` |
| Both new scope branches | Zero initialization verified; finite, nonzero gradients |
| Actual PaddyTrainer mini run | One CPU epoch on two synthetic images passed |
| Two-worker CPU DDP smoke check | Two epochs of forward/backward, validation loss and callbacks passed on both workers |
| Focused architecture/training/package/stability tests | 37 passed |

The verification output is `data/dysample_plus_verification/verification.json`. FLOPs remain unsupported because THOP does not account reliably for the functional sampling operations. Full-training validation accuracy is now available in [the fold 0 results report](dysample-plus-fold0-results.md); no GPU latency benchmark matching the earlier preflight has been performed for Plus.

The private [Kaggle experiment](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-dysample-plus-r2000-fold0), version 1, completed all 300 epochs and its outputs were retrieved on 2026-10-13 (Asia/Manila). This was a fresh R2000 fold 0 run initialized from `yolov8n.pt`. It used the same `configs/hyp_paper.yaml`, global batch 32, two GPUs, image size 640, AdamW, DIMIoU Strategy H, seed 42 and checkpoint period 50 as the completed plain DySample run. Outputs go to `/kaggle/working/dysample_training/proposed_plus/fold0`.

Build it with `python build_dysample_full_kernel.py --owner dmrashidferrer --fold 0 --variant proposed_plus`. Its default output folder is `kaggle_dysample_plus_kernel/`; the plain variant defaults to `kaggle_dysample_full_kernel/`. Each folder has its own notebook and Kaggle identity.

To resume an interrupted Plus run, mount its unstripped checkpoint and set `RESUME_FROM` in the Plus notebook. Resume validation checks the neck module arguments, including `dyscope=True`, variant output project, fold and training state. A checkpoint from plain DySample is rejected. Completed-run `best.pt` and `last.pt` may have optimizer/scaler state stripped, so preserve periodic checkpoints for continuation.

The completed plain DySample run's best saved validation mAP@50 and mAP@50-95 were 68.115% and 51.066%, respectively. DySample+ reached 65.314% and 46.667% on the same validation split, so enabling dynamic scope at both layers did not improve overall validation AP in this run. Both variants require a matched baseline and held-out test evaluation for the final architecture comparison. See [the results report](dysample-plus-fold0-results.md) for downloaded artifacts and the per-class comparison.
