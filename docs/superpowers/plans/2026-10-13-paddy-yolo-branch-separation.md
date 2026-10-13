# Separate Paddy-YOLO Replication into Dedicated Baseline Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, dedicated branch `Paddy-Yolo-Baseline` isolating the exact paper replication of Paddy-YOLO (removing experimental EMA attention modules, configs, tests, and kernels), and clean up baseline-only artifacts on `main` so `main` serves as the enhanced/EMA experiment branch.

**Architecture:** Create branch `Paddy-Yolo-Baseline` from current HEAD. On `Paddy-Yolo-Baseline`, remove all EMA extension files (`paddy_yolo/modules/ema.py`, `configs/paddy-yolo-ema.yaml`, `configs/paddy-yolo-p2-ema.yaml`, `configs/hyp_finetune.yaml`, `build_5fold_ema_kernel.py`, `build_fold0_ema_kernel.py`, `kaggle_5fold_ema_kernel/`, `kaggle_fold0_ema_kernel/`, `data/kaggle_ema_fold0_results/`, `tests/test_ema.py`, `tests/test_enhanced_model.py`, `tests/test_paddy_yolo_ema.py`), prune EMA references from `paddy_yolo/modules/register.py`, `paddy_yolo/modules/__init__.py`, and `verify_paddy_yolo.py`, and verify with the full 80 baseline test suite and 6-step paper verification suite. Then switch to `main` and remove baseline-only artifacts (`build_5fold_baseline_kernel.py`, `kaggle_5fold_baseline_kernel/`, `data/kaggle_baseline_5fold_results/`) and verify all 93 tests pass.

**Tech Stack:** Git, Python 3.13, PyTorch, Ultralytics YOLOv8, Pytest

## Global Constraints
- Preserve all existing paper recreation implementations: RGG module, GhostConv 5x5 DWConv, RepConv structural re-parameterization, DIMIoU Strategy H, R2000 dataset preparation.
- All 80 baseline unit tests must pass on `Paddy-Yolo-Baseline`.
- All 6 paper verification steps must pass on `Paddy-Yolo-Baseline`.
- All 93 tests must pass on `main`.
- Clean working directory with no untracked or dangling files.

---

### Task 1: Create `Paddy-Yolo-Baseline` Branch and Remove EMA Modules and Configs

**Files:**
- Modify: `paddy_yolo/modules/__init__.py`
- Modify: `paddy_yolo/modules/register.py`
- Delete: `paddy_yolo/modules/ema.py`
- Delete: `configs/paddy-yolo-ema.yaml`
- Delete: `configs/paddy-yolo-p2-ema.yaml`
- Delete: `configs/hyp_finetune.yaml`
- Delete: `build_5fold_ema_kernel.py`
- Delete: `build_fold0_ema_kernel.py`
- Delete directory: `kaggle_5fold_ema_kernel`
- Delete directory: `kaggle_fold0_ema_kernel`
- Delete directory: `data/kaggle_ema_fold0_results`
- Delete: `tests/test_ema.py`
- Delete: `tests/test_enhanced_model.py`
- Delete: `tests/test_paddy_yolo_ema.py`

**Interfaces:**
- Consumes: Ultralytics module registry
- Produces: Clean `register_paddy_modules()` registering only RGG, RGGBottleneck, RGGConv, RepConv, GhostConv.

- [ ] **Step 1: Create and switch to branch `Paddy-Yolo-Baseline`**

Run:
```bash
git checkout -b Paddy-Yolo-Baseline
```

- [ ] **Step 2: Delete EMA kernels, configs, and test files**

Remove:
```bash
git rm -rf build_5fold_ema_kernel.py build_fold0_ema_kernel.py kaggle_5fold_ema_kernel kaggle_fold0_ema_kernel configs/paddy-yolo-ema.yaml configs/paddy-yolo-p2-ema.yaml configs/hyp_finetune.yaml paddy_yolo/modules/ema.py tests/test_ema.py tests/test_enhanced_model.py tests/test_paddy_yolo_ema.py
```
And remove any untracked or cached results in `data/kaggle_ema_fold0_results`:
```powershell
Remove-Item -Recurse -Force data\kaggle_ema_fold0_results -ErrorAction SilentlyContinue
```

- [ ] **Step 3: Update `paddy_yolo/modules/__init__.py`**

Remove EMA export so `paddy_yolo/modules/__init__.py` is:
```python
"""Paddy-YOLO custom neural network modules."""

from paddy_yolo.modules.ghostconv import GhostConv
from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.modules.repconv import RepConv
from paddy_yolo.modules.rgg import RGG, RGGBottleneck, RGGConv

__all__ = [
    "RepConv",
    "GhostConv",
    "RGGConv",
    "RGGBottleneck",
    "RGG",
    "register_paddy_modules",
]
```

- [ ] **Step 4: Update `paddy_yolo/modules/register.py`**

Remove EMA import and registry entry:
```python
from paddy_yolo.modules.ghostconv import GhostConv
from paddy_yolo.modules.repconv import RepConv
from paddy_yolo.modules.rgg import RGG, RGGBottleneck, RGGConv

# Registry of custom Paddy-YOLO modules to inject
CUSTOM_MODULES: Dict[str, Type[nn.Module]] = {
    "RGG": RGG,
    "RGGBottleneck": RGGBottleneck,
    "RGGConv": RGGConv,
    "RepConv": RepConv,
    "GhostConv": GhostConv,
}
```

- [ ] **Step 5: Run pytest on `Paddy-Yolo-Baseline` to verify baseline unit tests**

Run: `pytest`
Expected: 80 passed in tests

---

### Task 2: Restore `verify_paddy_yolo.py` to 6 Paper Verification Steps on `Paddy-Yolo-Baseline`

**Files:**
- Modify: `verify_paddy_yolo.py`

**Interfaces:**
- Consumes: `paddy_yolo` baseline modules
- Produces: CLI `--smoke-test` and full verification checking paper Steps 1-6

- [ ] **Step 1: Remove EMA verification steps from `verify_paddy_yolo.py`**

Remove `from paddy_yolo.modules.ema import EMA`, `verify_ema()`, `verify_enhanced_model()`, and checks 7 and 8 from `run_verification()`. Update total check count message to 6 checks.

- [ ] **Step 2: Run verification script**

Run: `python verify_paddy_yolo.py --smoke-test`
Expected: Output ends with:
`Verification PASSED: All 6 verification checks completed successfully!`
Exit code: 0

- [ ] **Step 3: Run pytest test_scripts to ensure test_verify_script passes**

Run: `pytest tests/test_scripts.py`
Expected: 10 passed

- [ ] **Step 4: Commit all changes on `Paddy-Yolo-Baseline`**

Run:
```bash
git add .
git commit -m "chore: isolate pure Paddy-YOLO paper baseline replication"
```

---

### Task 3: Clean Up Baseline-Only Artifacts on `main` Branch

**Files:**
- Delete from `main`: `build_5fold_baseline_kernel.py`
- Delete directory from `main`: `kaggle_5fold_baseline_kernel`
- Delete directory from `main`: `data/kaggle_baseline_5fold_results`

**Interfaces:**
- Consumes: `main` branch with enhanced EMA models
- Produces: Clean `main` branch focused on EMA and architectural enhancements

- [ ] **Step 1: Switch back to `main` branch**

Run:
```bash
git checkout main
```

- [ ] **Step 2: Remove baseline-only artifacts from `main`**

Run:
```bash
git rm -rf build_5fold_baseline_kernel.py kaggle_5fold_baseline_kernel
```
And remove `data/kaggle_baseline_5fold_results` if untracked/committed:
```powershell
Remove-Item -Recurse -Force data\kaggle_baseline_5fold_results -ErrorAction SilentlyContinue
```

- [ ] **Step 3: Run full pytest suite on `main`**

Run: `pytest`
Expected: 93 passed in tests (all baseline + 13 EMA tests)

- [ ] **Step 4: Run verification suite on `main`**

Run: `python verify_paddy_yolo.py --smoke-test`
Expected: All 8 verification checks pass.

- [ ] **Step 5: Commit changes on `main`**

Run:
```bash
git add .
git commit -m "chore: clean up baseline-only artifacts to isolate enhanced experiments"
```

---

### Task 4: Verify Both Branches and Working Tree Cleanliness

**Files:**
- Inspect: git status, git branch -a, git log

- [ ] **Step 1: Verify git status on `main`**

Run:
```bash
git status
```
Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: Verify git status and tests on `Paddy-Yolo-Baseline`**

Run:
```bash
git checkout Paddy-Yolo-Baseline
git status
pytest
git checkout main
```
Expected: Clean working tree on both branches, 80 tests passing on `Paddy-Yolo-Baseline`, 93 tests passing on `main`.
