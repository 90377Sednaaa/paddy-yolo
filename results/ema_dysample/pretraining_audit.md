# Pretraining audit of the completed Kaggle run

The original initialization checkpoint and original `yolov8n.pt` were retrieved from notebook version 2 of `dmrashidpferrer/paddy-yolo-ema-dysample-fold0`. Every one of the **283 mapped state tensors matches its source exactly**. The source file also has the same SHA-256 checksum as the locally available `yolov8n.pt`.

The transfer is partial because this model replaces YOLOv8 C2f backbone bottlenecks with RGG and adds EMA and DySample. Compatible weights cover **2,398,848 of 3,388,416 parameter elements (70.80%)**, counting each parameter once. The 283/1471 state-entry ratio includes buffers and repeated aliases and does not measure parameter coverage.

| Model component | Total parameter elements | Pretrained parameter elements |
| --- | ---: | ---: |
| Backbone stride convolutions | 393,104 | 393,104 |
| RGG backbone blocks | 1,076,144 | 472,352 |
| SPPF | 164,608 | 164,608 |
| Existing neck C2f and Conv layers | 986,880 | 986,880 |
| Detect head | 754,432 | 381,904 |
| New EMA and DySample modules | 13,248 | 0 |
| Total | 3,388,416 | 2,398,848 |

RGG inherits its compatible projections and first bottleneck convolutions. Its new Rep/Ghost paths have no equivalent tensors in stock YOLOv8. EMA and DySample retain their own initialization. The Detect regression towers transfer; the classification towers do not, because COCO has 80 classes and the target has 16, changing both intermediate tower width (80 to 64) and final output width.

The original saved initialization was also loaded locally through `PaddyTrainer.setup_model()`. All 1471 state entries were preserved exactly by model construction and reload. The actual Kaggle training log records the same initialization file as the training model and reports loading 1471/1471 entries in the DDP worker. That second count means the complete mapped initialization was reloaded; it does not mean all 1471 entries originated in COCO pretraining.

## Performance concerns

The audit found no tensor-copying or shifted-layer mapping error. It does not establish that this transfer strategy is optimal for the changed architecture.

The run used AdamW with `lr0=0.01` and `warmup_bias_lr=0.1`. The first epoch's recorded bias-group learning rate was approximately 0.0708. Validation box, classification, and DFL losses were all `NaN` at epoch 1; validation classification loss reached 262.301 at epoch 3. All validation losses became finite from epoch 2 onward, and training completed 300 epochs. These observations show unstable early validation behavior, but do not establish its cause or quantify its effect on final accuracy.

Ultralytics [configuration guidance](https://docs.ultralytics.com/usage/cfg/) gives 0.001 as a starting rate for Adam, versus 0.01 for SGD. In the installed 8.3.18 trainer, automatic optimizer selection also sets bias warmup learning rate to 0.0. A lower AdamW learning rate and zero bias warmup are reasonable experiments, rather than proven fixes.

A controlled stock YOLOv8n run on the same dataset split is needed to establish whether the enhanced architecture underperforms that baseline. `results/baseline` currently contains architecture images, not trained baseline metrics. The best enhanced-model validation checkpoint was selected at epoch 266 according to the CSV's standard Ultralytics fitness (0.1 mAP50 + 0.9 mAP50-95); final test mAP50 is 59.03% and mAP50-95 is 41.75%.

## Evidence

- [Machine-readable audit](pretraining_audit.json): per-layer coverage, exact tensor mapping, skipped shapes, checksums, setup verification, and training snapshots.
- [Original COCO source checkpoint](initialization/yolov8n.pt).
- [Original Kaggle initialization](initialization/paddy-yolo-ema-dysample_init.pt).
- [Initialization artifact checksums](initialization/download_manifest.json).
- [Actual training log](training.log): initialization and DDP reload appear at the beginning.
- [Training history](results.csv): early validation instability and learning rates.

This audit downloaded and inspected existing artifacts and replayed model loading locally. It did not start another Kaggle training run.
