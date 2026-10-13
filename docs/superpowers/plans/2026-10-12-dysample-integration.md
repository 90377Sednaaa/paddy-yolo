# Paddy-YOLO DySample Integration Plan

> **For agentic workers:** Execute this plan in the current workspace; preserve the existing uncommitted baseline work.

**Goal:** Add an independently selectable Paddy-YOLO-DySample model and validate it before full Kaggle training.

**Architecture:** Keep `configs/paddy-yolo.yaml`, RGG, DIMIoU, three detection heads, and data definitions intact. Copy the upstream MIT DySample implementation into the project, teach the existing parser to inject the preceding channel count, and replace only neck layers 10 and 13 in a new YAML. Prepare equal pretrained initialization and separate Kaggle outputs for both variants.

**Tech Stack:** Python, PyTorch, Ultralytics 8.3.253, Kaggle GPU notebooks.

**Spec:** `C:/Users/danie/.codex/attachments/17399537-5487-4905-a5fd-60db5915f8fb/Pasted text.txt`

## Global Constraints

- DySample comes from `tiny-smart/dysample` under its MIT license; retain its offset, initialization, and grid sampling code.
- Use `scale=2`, `style='lp'`, `groups=4`, `dyscope=False`.
- Preserve the baseline YAML, RGG backbone, DIMIoU, detection heads, dataset and `configs/hyp_paper.yaml`.
- Both models use `yolov8n.pt`, total batch 32, 300 epochs, 640 pixels and AdamW for later full training.
- Prepare two T4 DDP when available, periodic checkpoints at 50 epochs, and explicit checkpoint resume.
- Run only small verification and profiling; do not start full training.

## Task 1: Module and parser

**Files:** Create `paddy_yolo/modules/dysample.py` and its MIT license copy; modify `paddy_yolo/modules/register.py` and module exports; create `tests/test_dysample.py`.

- [x] Add tests for exact scale/channel behavior, offset initialization, gradients and parser channel inference.
- [x] Confirm the tests fail before integration.
- [x] Copy upstream DySample exactly and preserve license and attribution.
- [x] Register DySample in normal and fallback parser paths with `args=[ch[f], *args]` and `c2=ch[f]`.
- [x] Run the focused tests.

## Task 2: Proposed YAML and pretrained initialization

**Files:** Create `configs/Paddy-YOLO-DySample.yaml`; extend `train.py` only for explicit periodic saving and resume path; create integration tests.

- [x] Duplicate the baseline YAML and replace only head layers 10 and 13 with DySample.
- [x] Test baseline/proposed YAML comparison, model build, P3/P4/P5 shapes and backward gradients.
- [x] Inspect and test the existing compatible-layer transfer from `yolov8n.pt` for both models; report loaded and unmatched layers.
- [x] Add opt-in `--save-period` and `--resume-from` while preserving existing defaults.

## Task 3: Kaggle preparation and measurements

**Files:** Create a small verification/profiling script and a private Kaggle notebook builder; document exact run commands and results.

- [x] Build a source-only, credential-free Kaggle package for `dmrashidferrer`.
- [x] Run a small test: initialization, forward, backward, feature concatenations, pretrained transfer and shape hooks.
- [x] Profile both models with identical 640-pixel FP32 inputs and one GPU; mark DySample FLOPs unsupported if `grid_sample` is uncounted.
- [x] Prepare distinct training output paths, two-GPU DDP when available, batch 32, save every 50 epochs and explicit resume.
- [x] Verify no full training starts automatically and report the measured numbers and limits.

