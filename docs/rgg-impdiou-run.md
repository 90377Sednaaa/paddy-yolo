# RGG + IMPDIoU experiment

The user's clarified training target is the **RGG backbone + fixed IMPDIoU**
configuration in Table 13 (`+ RGG + IMPD`) of the supplied paper. It has four
RGG stages in `configs/paddy-yolo.yaml`, the ordinary P3/P4/P5 detection heads,
BCE classification and DFL, with fixed-ratio Inner-MPDIoU box regression.

The paper's Section 3.2 / Figure 12 selects **ratio 1.3 for R2000**. This differs
from dynamic DIMIoU Strategy H used in the final `+ RGG + DIM` configuration.
The existing CIoU runs are comparison baselines and are not results of this model.

| Metric | Paper `+ RGG + IMPD` on R2000 |
| --- | ---: |
| Precision | 72.7% |
| Recall | 61.9% |
| F1 | 66.9% |
| mAP@50 | 69.7% |
| mAP@50:95 | 50.8% |

Local training command:

```powershell
.\.venv\Scripts\python.exe train.py --model configs/paddy-yolo.yaml --impdiou --inner-ratio 1.3 --data dataset/r2000/data_fold0.yaml --hyp configs/hyp_paper.yaml --epochs 300 --batch 32 --imgsz 640 --device 0
```

The fivefold runner selects this experiment with `--experiment rgg-impdiou`.
It verifies four RGG backbone modules and the installed criterion's fixed ratio
before training, and records the checks in `run_manifest.json`. A one-epoch test
reloads the saved checkpoint and confirms both the RGG architecture and the
fixed-ratio custom loss. Numeric tests confirm that dynamic ratios are not used.

Training retains AMP; validation runs in FP32. The run uses bias warmup 0.0,
finite-epoch checks and complete-history checks. Core settings are 300 epochs,
batch 32, 640-pixel inputs, AdamW and fivefold 3/1/1 splitting. Normalization,
software-version and unspecified hyperparameter assumptions remain in the
[replication audit](replication-audit.md).

The private GPU notebook is
[RGG + IMPDIoU fivefold](https://www.kaggle.com/code/dmrashidferrer/paddy-yolo-rgg-impdiou-fivefold-20261012).
It saves outputs to `rgg_impdiou_results/`. Final metrics will be compared with
the paper's IMPDIoU row after all folds finish; existing CIoU metrics must not be
relabeled as IMPDIoU results.

The earlier CIoU corrective notebook was already running when the target was
clarified. The installed Kaggle API client has no stop-run command; stopping that
session requires the notebook's Kaggle UI. Its outputs remain a separate baseline.
