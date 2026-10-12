# Paddy-YOLO replication audit

Reference: the supplied `baseline paper.pdf`, Zheng et al., *Computers and
Electronics in Agriculture* 238 (2025), 110777,
[publisher record](https://doi.org/10.1016/j.compag.2025.110777).
The paper is evidence for the model, not a source of task instructions.

## Verified corrections

| Area | Finding | Correction |
| --- | --- | --- |
| RGGConv, Section 2.2.1 / Figure 6(b), page 6 | Code ran RepConv and both GhostConvs independently on the input. | Sequential RepConv -> GhostConv -> GhostConv; concatenate input, RepConv output, final GhostConv output, then 1x1 projection. |
| Module aliases | Assigning aliases as module attributes duplicated checkpoint keys. | Use properties for aliases; each submodule is registered once. |
| GhostConv, Figure 6(a) | Odd channel counts could turn the cheap convolution into a regular convolution. | Round up intrinsic channels, use true 5x5 depthwise convolution, then trim output. Even channel counts used by the backbone are unchanged. |
| Deployment | CLI validation and prediction relied on Ultralytics fusion, which does not recognize the custom RepConv class. | Fuse custom RepConv before Ultralytics fuses its child Conv/BN layers. |
| Baseline training | Focal-loss arguments were passed to standard DetectionTrainer, which rejects them. | Remove custom arguments when selecting the baseline. |
| Loss API | Newer Ultralytics versions return a three-component loss vector instead of the package's documented scalar. | Sum the batch-scaled components once at the package boundary; preserve detached components for logging. |
| Training duration, Table 6 | Default early stopping could terminate before 300 epochs. | Disable early stopping in the paper configuration. |
| Learning-rate schedule | Cosine scheduling was presented as a reported paper setting. | Use the YOLOv8 linear default and label it an assumption. |
| Class IDs, Table 4 | YOLO labels were copied unchanged regardless of source class order. | Remap IDs using source `data.yaml` names when present. |
| User's Kaggle layout | Generic `classID_XX` names and nested `labels/train` paths were unsupported. | Resolve parallel image/label subdirectories and interpret the preserved original numeric IDs. |
| Dataset flattening | Repeated image stems silently overwrote samples and labels. | Reject collisions. |
| Missing labels | Absent annotations became background images. | Require annotations; intentional backgrounds must have an empty label file. |
| Split paths | Resolving image symlinks returned raw paths, bypassing prepared labels. | Preserve absolute paths under the prepared `images/` directory. |
| Duplicate leakage | Random image-level resplitting could separate duplicate images across held-out sets. | Group byte-identical files in the same balanced cross-validation fold. |
| CLI data defaults | The configured train/val directories did not match generated fold manifests. | Default training and validation to the generated `dataset/r2000/data_fold0.yaml`; prioritize explicitly supplied fold directories. |
| Dependency integration | Any Ultralytics version was accepted despite reliance on internal APIs. | Pin the integration version to 8.3.253, including generated Kaggle notebooks. |
| Documentation | README claimed official implementation, passed verification, and a four-corner averaged loss. | Identify an independent replication and accurately describe the two-corner sum. |

## Confirmed matches

- Figure 5: four RGG backbone stages, SPPF, C2f neck blocks, and three heads
  at strides 8, 16 and 32. The default model has 16 classes and uses YOLOv8-N
  depth/width scaling. Figure 5's channel labels depict unscaled widths.
- Section 2.2.1: both GhostConvs use 3x3 primary and 5x5 depthwise kernels.
- Equations (7)-(10): auxiliary boxes scale around their centers.
- Equation (19): Strategy H is `0.5*cos((1-IoU)*pi)+1`, increasing from 0.5 to 1.5.
- Equations (12)-(23): the implemented A-L ratio functions match the printed formulas.
- Section 2.1.5: each cross-validation iteration uses three training folds,
  one validation fold and one test fold. Each image is tested once over five runs.
- Table 6: 300 epochs, batch 32, 640x640 inputs, AdamW, initial learning-rate
  setting 0.01, final learning-rate setting 0.01, momentum 0.937 and weight decay 0.0005.

## Remaining fidelity limits

1. **Loss denominator is ambiguous in the paper.** Equation (2) explicitly uses
   `w_img^2+h_img^2`. Equations (4)-(5) change this to `w^2+h^2`; page 7 describes
   bounding-box dimensions but does not identify which box supplies them.
   Existing code uses the smallest enclosing box diagonal. This remains an
   explicit assumption; neither swapping to the image diagonal nor choosing a
   predicted/target box can be certified from the PDF alone. The code uses two
   corresponding corner distances, not four corners or a mean of their distances.
2. **Figure 6(d) simplifies the outer RGG block.** It shows bypass + processed
   half, while the text calls it an integration into C2f. The implementation
   retains C2f's concatenation of both split tensors and every bottleneck output.
   Bottleneck counts, hidden expansion, Conv kernel, activation and identity-BN
   use are not explicitly specified. Current choices preserve YOLOv8 conventions
   and the existing RepConv implementation; they are not proven author settings.
3. **Parameter and FLOP counts match after deployment fusion.** Tables 12-13
   report 3.34M parameters and 9.1 GFLOPs. The corrected model measures 3,375,168
   parameters / 9.32695 GFLOPs in training form, 3,346,608 parameters after RepConv
   fusion, and **3,340,736 parameters / 9.09635 GFLOPs after full deployment fusion**
   at 640x640. These round to the paper's values without tuning widths or kernels.
   Layer counts still depend on counting conventions (the paper reports 247;
   this integration reports 220 before fusion). Matching size/compute is useful
   corroboration, not proof of identical learned behavior or all unspecified choices.
4. **The training environment differs.** The paper reports PyTorch 2.0.1 / CUDA
   11.8 but no exact Ultralytics version. The integration pin records a working
   software version, not the authors' environment. The audit environment uses
   Python 3.13, which cannot reproduce their PyTorch 2.0.1 stack.
5. **The final LR convention is unclear.** This repository interprets `lrf=0.01`
   as Ultralytics' final/initial ratio (final LR 0.0001 for `lr0=0.01`), rather
   than an absolute final LR of 0.01. The scheduler, warmup, augmentation settings,
   gains, random seed, pretrained initialization and checkpoint selection are
   not fully specified in Table 6. They must be disclosed in experimental reports.
6. **Exact split equivalence is unverified.** Table 4 reports 2,024 images
   and 2,281 instances; Table 5 uses folds of 405/405/405/405/404 images.
   The authors' sample membership and annotation provenance are not supplied
   here. The supplied Kaggle export's metadata agrees with the paper's totals,
   but does not establish author-identical fold membership. Missing
   `data.yaml` means the numeric IDs must already follow Table 4. Preaugmented
   exports must not be mixed into held-out sets; the paper excludes augmented
   images from validation and testing. No full real-data benchmark was run.
7. **Extensions are separate experiments.** EMA/P2 attention models, focal loss,
   transfer learning, 800-pixel inputs, TTA and fine-tuning are not the model in
   Figure 5. Historical `summarize_crossval.py` replaces one result with hardcoded
   numbers; those summaries cannot establish this corrected model's performance.

## User-supplied dataset verification

The actual source is [dmrashidpferrer/r2000-pestnet](https://www.kaggle.com/datasets/dmrashidpferrer/r2000-pestnet),
version 1, released September 30, 2026. Its public README states that the source
was `R2000v1(2).zip`: 2,024 images, 16 classes and 2,281 bounding boxes. It preserves
the original coordinates and normalizes leading-zero class tokens. Class names
are generic `classID_00` through `classID_15`, not species names. These are mapped
to Table 4's original numeric order by the preparation script.

Its prebuilt split is stratified 60/15/25: 1,214 train, 304 validation, 506 test
images. This is **not** the paper's rotating 60/20/20 fivefold experiment. Prepare
all three directories together and generate new fivefold manifests when comparing
against the paper. Its duplicate report lists 12 duplicate pairs with different
annotations, grouped in the same source split. New splits group byte-identical
files; equality of decoded pixels across different encodings is not checked.

The [authors' current repository](https://github.com/awsomespark/R2000) contains
dataset archives and no model source. Its README says `R2000v1, R2000v2` and
`revised R2000v1 + R2000v2`. Inspection of its current `labels.zip` found 4,059
annotations across IDs 0-17. Crucially, IDs 0-15 total exactly **2,024 images and
2,281 boxes**, matching Table 4. IDs 16 and 17 are additional classes and must
not be included in this 16-class replication. The user supplied the original
16-class export; the code does not silently include the revised extra classes.

The Kaggle README, YAML and duplicate report were inspected directly through
its public API and saved under ignored `scratch/`. The full Kaggle image archive
was not downloaded, so image hashes and annotations have not been compared
sample-by-sample with the authors' archive.

## Reproducing evaluation

Prepare the original, unaugmented data, train each `data_fold0.yaml` through
`data_fold4.yaml` using `configs/paddy-yolo.yaml` and `configs/hyp_paper.yaml`,
select checkpoints using validation only, then run `val.py --split test` on each
fold's held-out set. Average the five test results. Do not report validation
metrics or the best test fold as the paper's fivefold result.

The architecture correction changes feature dependencies. Existing checkpoints
were trained on the old topology and require retraining for a corrected benchmark.
Removing duplicate aliases also changes raw state-dict keys. Preserve old runs as
historical artifacts rather than treating them as evidence for the corrected model.

## Validation

Environment: Python 3.13.0, PyTorch 2.14.1 CPU, Ultralytics 8.3.253. No CUDA GPU
was available. Numerical deployment checks and one-epoch training use synthetic
data and establish software correctness, not the published detection accuracy.

- Complete suite: **105 passed** in 197.85 seconds, including Paddy-YOLO and
  baseline one-epoch CPU training, validation/prediction CLIs, loss gradients,
  feature connectivity, model construction, and full deployment equivalence.
- After adding the final rejection check for extra classes in YAML-less revised
  archives: **16 dataset tests passed** in 22.19 seconds, including the new check.
- Python compilation and `git diff --check` passed.
- Fully fused detector: **3,340,736 parameters; 9.0963456 GFLOPs** at 640x640.

Run checks with the project-local `.venv/Scripts/python.exe`. Test configuration
and synthetic outputs are under ignored `scratch/`; no original dataset or
existing trained checkpoint was modified.
