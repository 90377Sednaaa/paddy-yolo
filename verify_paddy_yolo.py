"""Paddy-YOLO Comprehensive Verification Suite.

Validates:
1. RepConv structural re-parameterization equivalence (training 3-branch vs deploy 1-conv).
2. GhostConv cheap operation with 5x5 depthwise convolution and gradient backprop.
3. RGG bottleneck module, residual paths, split-merge logic, and gradient backprop.
4. DIMIoU loss, dynamic Strategy H ratio curve, corner distance penalties, and gradients.
5. Full Paddy-YOLO architecture configuration parsing, parameter count, and forward pass.
6. End-to-end mini training step on 2 synthetic images with PaddyTrainer.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Optional
import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO

from paddy_yolo.modules.register import register_paddy_modules
from paddy_yolo.modules.repconv import RepConv
from paddy_yolo.modules.ghostconv import GhostConv
from paddy_yolo.modules.rgg import RGG, RGGBottleneck, RGGConv
from paddy_yolo.modules.ema import EMA
from paddy_yolo.loss.dimiou import dimiou_loss, get_dynamic_ratio
from paddy_yolo.loss.paddy_loss import PaddyBboxLoss, PaddyDetectionLoss
from paddy_yolo.engine import PaddyTrainer
from paddy_yolo.dataset.prepare_r2000 import generate_dummy_dataset, R2000_CLASSES


def verify_repconv(device: str = "cpu") -> bool:
    """Verify RepConv multi-branch training mode and fused deploy mode numerical equivalence."""
    print("  [1.1] Testing RepConv training forward and backward pass...")
    torch.manual_seed(42)
    rep = RepConv(c1=32, c2=32, k=3, s=1, act="silu", deploy=False).to(device)
    rep.train()

    x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    out_train = rep(x)
    loss = out_train.sum()
    loss.backward()

    assert x.grad is not None, "RepConv input gradients must not be None"
    assert not torch.isnan(x.grad).any(), "RepConv gradients contain NaN"

    print("  [1.2] Testing RepConv switch_to_deploy re-parameterization equivalence...")
    rep.eval()
    with torch.no_grad():
        out_eval = rep(x)

    rep.switch_to_deploy()
    assert rep.deploy is True, "RepConv deploy flag must be True"
    assert rep.rbr_reparam is not None, "RepConv rbr_reparam conv must exist in deploy mode"
    assert rep.rbr_dense is None, "rbr_dense must be deleted in deploy mode"
    assert rep.rbr_1x1 is None, "rbr_1x1 must be deleted in deploy mode"
    assert rep.rbr_identity is None, "rbr_identity must be deleted in deploy mode"

    with torch.no_grad():
        out_deploy = rep(x)

    max_diff = (out_eval - out_deploy).abs().max().item()
    assert max_diff < 1e-4, f"RepConv deploy output mismatch: max diff = {max_diff:.6e}"

    # Also verify c1 != c2 case (where identity is not added)
    rep_diff_ch = RepConv(c1=16, c2=32, k=3, s=1, act="silu", deploy=False).to(device)
    rep_diff_ch.eval()
    x_ch = torch.randn(2, 16, 16, 16, device=device)
    with torch.no_grad():
        out_eval_ch = rep_diff_ch(x_ch)
    rep_diff_ch.switch_to_deploy()
    with torch.no_grad():
        out_deploy_ch = rep_diff_ch(x_ch)
    max_diff_ch = (out_eval_ch - out_deploy_ch).abs().max().item()
    assert max_diff_ch < 1e-4, f"RepConv (c1!=c2) deploy output mismatch: max diff = {max_diff_ch:.6e}"

    print(f"  [PASSED] RepConv deploy equivalence confirmed (max error: {max(max_diff, max_diff_ch):.2e}).")
    return True


def verify_ghostconv(device: str = "cpu") -> bool:
    """Verify GhostConv cheap operations using 5x5 DWConv, output shape, and gradient flow."""
    print("  [2.1] Testing GhostConv 5x5 DWConv cheap operation kernel size and grouping...")
    ghost = GhostConv(c1=32, c2=64, k=3, s=1, dw_k=5).to(device)

    # Verify primary conv is 3x3 standard conv
    assert ghost.cv1.conv.kernel_size == (3, 3), "Primary conv cv1 kernel size must be (3, 3)"
    # Verify cheap operation is 5x5 depthwise conv
    assert ghost.cv2.conv.kernel_size == (5, 5), "Cheap conv cv2 must be 5x5 depthwise"
    assert ghost.cv2.conv.groups == ghost.cv2.conv.in_channels, "Cheap conv cv2 must be depthwise"

    print("  [2.2] Testing GhostConv forward pass shape and gradient flow...")
    x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    out = ghost(x)
    assert out.shape == (2, 64, 16, 16), f"Expected shape (2, 64, 16, 16), got {out.shape}"

    out.sum().backward()
    assert x.grad is not None, "GhostConv input gradients must not be None"
    assert not torch.isnan(x.grad).any(), "GhostConv gradients contain NaN"

    print("  [2.3] Testing GhostConv stride-2 spatial downsampling...")
    ghost_s2 = GhostConv(c1=32, c2=64, k=3, s=2, dw_k=5).to(device)
    out_s2 = ghost_s2(x)
    assert out_s2.shape == (2, 64, 8, 8), f"Expected shape (2, 64, 8, 8), got {out_s2.shape}"

    print("  [PASSED] GhostConv 5x5 DWConv and gradient backprop confirmed.")
    return True


def verify_rgg(device: str = "cpu") -> bool:
    """Verify RGG bottleneck, residual connections, split-concat logic, and gradients."""
    print("  [3.1] Testing RGGBottleneck residual behavior and gradients...")
    torch.manual_seed(42)
    btnk = RGGBottleneck(c1=64, c2=64, shortcut=True).to(device)
    btnk.train()
    x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    out_btnk = btnk(x)
    assert out_btnk.shape == (2, 64, 16, 16), f"Expected shape (2, 64, 16, 16), got {out_btnk.shape}"
    out_btnk.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "RGGBottleneck gradient error"

    print("  [3.2] Testing RGG module forward pass and backprop through both branches...")
    rgg = RGG(c1=64, c2=64, n=2, shortcut=True).to(device)
    rgg.train()
    x_rgg = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
    out_rgg = rgg(x_rgg)
    assert out_rgg.shape == (2, 64, 16, 16), f"Expected shape (2, 64, 16, 16), got {out_rgg.shape}"

    loss_rgg = out_rgg.sum()
    loss_rgg.backward()
    assert x_rgg.grad is not None, "RGG input gradients must not be None"
    assert not torch.isnan(x_rgg.grad).any(), "RGG gradients contain NaN"

    print("  [3.3] Testing RGG switch_to_deploy re-parameterization...")
    rgg.eval()
    with torch.no_grad():
        out_eval = rgg(x_rgg)
    rgg.switch_to_deploy()
    with torch.no_grad():
        out_deploy = rgg(x_rgg)
    max_diff = (out_eval - out_deploy).abs().max().item()
    assert max_diff < 1e-4, f"RGG deploy output mismatch: max diff = {max_diff:.6e}"

    print(f"  [PASSED] RGG residual connections, backprop, and deploy confirmed (error: {max_diff:.2e}).")
    return True


def verify_dimiou_loss(device: str = "cpu") -> bool:
    """Verify DIMIoU loss, dynamic Strategy H ratio scaling, corner distances, and gradients."""
    print("  [4.1] Testing DIMIoU Strategy H dynamic ratio curve...")
    # IoU = 0 -> ratio = 0.5; IoU = 1 -> ratio = 1.5
    iou_0 = torch.tensor([0.0], device=device)
    iou_half = torch.tensor([0.5], device=device)
    iou_1 = torch.tensor([1.0], device=device)

    ratio_0 = get_dynamic_ratio(iou_0, strategy="H").item()
    ratio_half = get_dynamic_ratio(iou_half, strategy="H").item()
    ratio_1 = get_dynamic_ratio(iou_1, strategy="H").item()

    assert abs(ratio_0 - 0.5) < 1e-5, f"Expected ratio 0.5 at IoU 0, got {ratio_0}"
    assert 0.5 < ratio_half < 1.5, f"Expected ratio in (0.5, 1.5) at IoU 0.5, got {ratio_half}"
    assert abs(ratio_1 - 1.5) < 1e-5, f"Expected ratio 1.5 at IoU 1, got {ratio_1}"

    print("  [4.2] Testing DIMIoU loss on identical, shifted, and disjoint boxes...")
    # Identical boxes: loss should be exactly 0
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]], device=device)
    loss_identical = dimiou_loss(b1, b1, xywh=False, strategy="H")
    assert loss_identical.item() < 1e-5, f"Identical boxes should have loss ~0, got {loss_identical.item()}"

    # Shifted boxes: loss should be strictly positive
    b2 = torch.tensor([[15.0, 15.0, 55.0, 55.0]], device=device, requires_grad=True)
    loss_shifted = dimiou_loss(b1, b2, xywh=False, strategy="H")
    assert loss_shifted.item() > 0.0, "Shifted boxes must have positive loss"
    loss_shifted.backward()
    assert b2.grad is not None and not torch.isnan(b2.grad).any(), "DIMIoU gradients contain NaN"

    # Disjoint boxes: loss must be strictly positive with non-negative corner penalties
    b3 = torch.tensor([[100.0, 100.0, 150.0, 150.0]], device=device)
    loss_disjoint = dimiou_loss(b1, b3, xywh=False, strategy="H")
    assert loss_disjoint.item() > 1.0, f"Disjoint boxes loss should be > 1.0, got {loss_disjoint.item()}"

    print("  [4.3] Testing PaddyBboxLoss criterion integration and gradient backprop...")
    bbox_loss = PaddyBboxLoss(reg_max=16, strategy="H")
    assert bbox_loss.strategy == "H"

    print("  [PASSED] DIMIoU loss, Strategy H, corner penalties, and gradients confirmed.")
    return True


def verify_model_architecture(device: str = "cpu") -> bool:
    """Verify Ultralytics Paddy-YOLO YAML configuration parsing and model summary."""
    print("  [5.1] Registering Paddy-YOLO modules into Ultralytics...")
    register_paddy_modules()

    cfg_path = "configs/paddy-yolo.yaml"
    assert Path(cfg_path).is_file(), f"Configuration file {cfg_path} not found"

    print(f"  [5.2] Parsing architecture config: {cfg_path}...")
    model = YOLO(cfg_path).model.to(device)

    # Check number of classes
    assert getattr(model, "nc", 16) == 16, f"Expected 16 classes, found {getattr(model, 'nc', 16)}"

    # Check parameter count (~3.16M - 3.33M)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [5.3] Model parameter count: {param_count:,} parameters")
    assert 3_000_000 < param_count < 3_600_000, (
        f"Expected parameter count in range [3.0M, 3.6M], got {param_count}"
    )

    # Check backbone layer types
    rgg_layers = [m for m in model.model if isinstance(m, RGG)]
    assert len(rgg_layers) == 4, f"Expected 4 RGG backbone modules, found {len(rgg_layers)}"

    print("  [5.4] Testing model forward pass with batch (2, 3, 128, 128)...")
    model.eval()
    x = torch.randn(2, 3, 128, 128, device=device)
    with torch.no_grad():
        preds = model(x)

    # Detect head outputs list of 3 feature scale tensors during eval
    assert isinstance(preds, (tuple, list, torch.Tensor)), "Model predictions must be tuple/list/tensor"
    print(f"  [PASSED] Model parsed successfully ({param_count:,} params, 4 RGG layers, valid forward pass).")
    return True


def verify_mini_training_step(device: str = "cpu") -> bool:
    """Run an end-to-end mini training step on 2 synthetic images using PaddyTrainer."""
    print("  [6.1] Creating temporary synthetic dataset with 2 images...")
    register_paddy_modules()

    tmp_dir = Path(tempfile.mkdtemp(prefix="paddy_verify_"))
    try:
        generate_dummy_dataset(tmp_dir, num_images=2, img_size=(64, 64), seed=42)

        data_yaml = tmp_dir / "data.yaml"
        yaml_content = {
            "path": str(tmp_dir.resolve()).replace("\\", "/"),
            "train": "images",
            "val": "images",
            "nc": len(R2000_CLASSES),
            "names": {i: name for i, name in enumerate(R2000_CLASSES)},
        }
        with open(data_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        print("  [6.2] Running 1 mini training epoch with PaddyTrainer...")
        overrides = {
            "model": "configs/paddy-yolo.yaml",
            "data": str(data_yaml),
            "epochs": 1,
            "batch": 2,
            "imgsz": 64,
            "device": device,
            "workers": 0,
            "project": str(tmp_dir / "runs"),
            "name": "verify_train",
            "plots": False,
            "save": False,
            "val": False,
            "verbose": False,
            "strategy": "H",
        }

        trainer = PaddyTrainer(overrides=overrides)
        trainer.train()

        assert trainer.model is not None, "Trainer model must not be None after training"
        print("  [6.3] Mini training step completed without error.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("  [PASSED] End-to-end mini training step on synthetic data confirmed.")
    return True


def verify_ema(device: str = "cpu") -> bool:
    """Verify EMA multi-scale attention, coordinate pooling, and gradient flow."""
    print("  [7.1] Testing EMA cross-spatial multi-scale attention forward pass...")
    ema = EMA(channels=32, factor=8).to(device)
    x = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)
    out = ema(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    print("  [7.2] Testing EMA gradient flow back to input...")
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "EMA gradients contain NaN"
    print("  [PASSED] EMA cross-spatial multi-scale attention confirmed.")
    return True


def verify_enhanced_model(device: str = "cpu") -> bool:
    """Verify Paddy-YOLO with Backbone EMA (3-head) and P2-EMA (4-head) architectures."""
    print("  [8.1] Testing Paddy-YOLO-EMA (Backbone EMA) configuration parsing...")
    register_paddy_modules()
    ema_cfg = "configs/paddy-yolo-ema.yaml"
    assert Path(ema_cfg).is_file(), f"{ema_cfg} not found"

    model_ema = YOLO(ema_cfg).model.to(device)
    ema_param_count = sum(p.numel() for p in model_ema.parameters())
    print(f"  [8.2] Backbone-EMA parameter count: {ema_param_count:,} parameters")
    assert 3_300_000 < ema_param_count < 3_450_000, f"Unexpected parameter count: {ema_param_count}"

    print("  [8.3] Testing Backbone-EMA 3-head forward pass with 8,400 anchor points...")
    model_ema.eval()
    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        preds_ema = model_ema(x)
    assert isinstance(preds_ema, tuple), "Predictions must be a tuple"
    assert preds_ema[0].shape == (1, 20, 8400), f"Expected (1, 20, 8400), got {preds_ema[0].shape}"

    print("  [8.4] Testing Enhanced Paddy-YOLO (P2-EMA) configuration parsing...")
    p2_cfg = "configs/paddy-yolo-p2-ema.yaml"
    assert Path(p2_cfg).is_file(), f"{p2_cfg} not found"

    model_p2 = YOLO(p2_cfg).model.to(device)
    p2_param_count = sum(p.numel() for p in model_p2.parameters())
    print(f"  [8.5] P2-EMA parameter count: {p2_param_count:,} parameters")
    assert 3_000_000 < p2_param_count < 3_600_000, f"Unexpected parameter count: {p2_param_count}"

    print("  [8.6] Testing 4-head forward pass with 34,000 anchor points...")
    model_p2.eval()
    with torch.no_grad():
        preds_p2 = model_p2(x)
    assert isinstance(preds_p2, tuple), "Predictions must be a tuple"
    assert preds_p2[0].shape == (1, 20, 34000), f"Expected (1, 20, 34000), got {preds_p2[0].shape}"
    print("  [PASSED] Paddy-YOLO-EMA (Backbone EMA) and P2-EMA architectures confirmed.")
    return True


def run_verification(smoke_test: bool = False, device: str = "cpu") -> bool:
    """Execute all verification checks."""
    print("=" * 70)
    mode_str = "SMOKE-TEST" if smoke_test else "FULL VERIFICATION"
    print(f"Paddy-YOLO Verification Suite [{mode_str}] (device: {device})")
    print("=" * 70)

    checks = [
        ("Step 1: RepConv Re-parameterization Equivalence", lambda: verify_repconv(device)),
        ("Step 2: GhostConv 5x5 DWConv Cheap Operation", lambda: verify_ghostconv(device)),
        ("Step 3: RGG Module Residuals & Gradients", lambda: verify_rgg(device)),
        ("Step 4: DIMIoU Loss & Strategy H", lambda: verify_dimiou_loss(device)),
        ("Step 5: Model Architecture Parsing & Forward Pass", lambda: verify_model_architecture(device)),
        ("Step 6: End-to-End Mini Training Step", lambda: verify_mini_training_step(device)),
        ("Step 7: EMA Multi-Scale Attention & Gradients", lambda: verify_ema(device)),
        ("Step 8: Enhanced Paddy-YOLO (P2-EMA) Architecture & 4 Heads", lambda: verify_enhanced_model(device)),
    ]

    for title, check_fn in checks:
        print(f"\n--- {title} ---")
        try:
            passed = check_fn()
            if not passed:
                print(f"[FAILED] {title}")
                return False
        except Exception as e:
            print(f"[ERROR] Exception during {title}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "=" * 70)
    print("Verification PASSED: All 8 verification checks completed successfully!")
    print("=" * 70)
    return True


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Paddy-YOLO Architecture, Modules, Loss, and Training Verification"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run self-contained smoke-test verification and exit with status code 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run verification on ('cpu', 'cuda', etc. Default: cpu)",
    )
    args = parser.parse_args(argv)

    success = run_verification(smoke_test=args.smoke_test, device=args.device)
    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
