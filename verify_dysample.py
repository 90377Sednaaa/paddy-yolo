"""Small baseline/DySample architecture, transfer, and complexity comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from paddy_yolo.dataset.prepare_r2000 import R2000_CLASSES, generate_dummy_dataset
from paddy_yolo.modules.dysample import DySample
from paddy_yolo.modules.rgg import RGG


MODELS = {
    "baseline": "configs/paddy-yolo.yaml",
    "Paddy-YOLO-DySample": "configs/Paddy-YOLO-DySample.yaml",
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inspect_transfer(model: torch.nn.Module, source: torch.nn.Module) -> dict:
    """Load by Ultralytics' normal name/shape matching and audit actual tensors."""
    source_state = source.state_dict()
    before = model.state_dict()
    compatible = {key for key, value in before.items()
                  if key in source_state and value.shape == source_state[key].shape}
    model.load(source)
    after = model.state_dict()
    loaded = sorted(key for key in compatible if torch.equal(after[key], source_state[key]))
    assert "model.0.conv.weight" in loaded, "Expected pretrained first convolution was not loaded"
    assert len(loaded) == len(compatible), "Not every shape-compatible tensor was transferred"
    initialized = sorted(set(after) - compatible)
    return {
        "source": "yolov8n.pt",
        "loaded_tensor_count": len(loaded),
        "initialized_tensor_count": len(initialized),
        "loaded_tensor_keys": loaded,
        "initialized_tensor_keys": initialized,
        "loaded_layer_paths": sorted({key.rsplit(".", 1)[0] for key in loaded}),
        "initialized_layer_paths": sorted({key.rsplit(".", 1)[0] for key in initialized}),
    }


def measure(model: torch.nn.Module, image: torch.Tensor, runs: int = 15) -> dict:
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(image)
        synchronize(image.device)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(image)
            synchronize(image.device)
            times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(times), "runs": runs}


def inspect_model(name: str, config: str, source: torch.nn.Module,
                  image: torch.Tensor, output: Path) -> dict:
    model = YOLO(config).model.to(image.device)
    layers = model.model
    assert [index for index, layer in enumerate(layers) if isinstance(layer, RGG)] == [2, 4, 6, 8]
    if name == "baseline":
        assert not any(isinstance(layer, DySample) for layer in layers)
    else:
        assert [index for index, layer in enumerate(layers) if isinstance(layer, DySample)] == [10, 13]
        assert [(layers[index].offset.in_channels, layers[index].scale, layers[index].groups)
                for index in (10, 13)] == [(256, 2, 4), (128, 2, 4)]

    transfer = inspect_transfer(model, source)
    shapes = []
    hook = layers[-1].register_forward_pre_hook(
        lambda _module, inputs: shapes.append([list(tensor.shape) for tensor in inputs[0]])
    )
    try:
        model.eval()
        with torch.inference_mode():
            model(image)
    finally:
        hook.remove()
    expected = [[1, 64, 80, 80], [1, 128, 40, 40], [1, 256, 20, 20]]
    assert shapes[-1] == expected, f"{name} detection inputs: {shapes[-1]}"
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint = output / f"{name}_fp32_state_dict.pt"
    torch.save(model.state_dict(), checkpoint)
    result = {
        "config": config,
        "trainable_parameters": params,
        "top_level_layers": len(layers),
        "module_count": sum(1 for _ in model.modules()),
        "detection_input_shapes": shapes[-1],
        "serialized_fp32_state_dict_bytes": checkpoint.stat().st_size,
        "pretrained_transfer": transfer,
        "inference": measure(model, image),
    }
    if name == "baseline":
        result["flops_g"] = get_flops(model, imgsz=640)
        result["flops_method"] = "Ultralytics/THOP estimate at 640px"
    else:
        result["flops_g"] = None
        result["flops_method"] = (
            "Unsupported: DySample uses functional pixel_shuffle and grid_sample; "
            "THOP does not count these sampling operations reliably."
        )
    return result


def run_small_training(config: str, name: str, weights: str,
                       device: torch.device, output: Path,
                       device_override: str | None = None) -> dict:
    """Exercise the actual training CLI for one epoch on two synthetic images."""
    dataset = output / "mini_dataset"
    if not dataset.exists():
        generate_dummy_dataset(dataset, num_images=2, img_size=(64, 64), seed=42)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": dataset.resolve().as_posix(), "train": "images", "val": "images",
        "nc": len(R2000_CLASSES),
        "names": {index: label for index, label in enumerate(R2000_CLASSES)},
    }), encoding="utf-8")
    command = [sys.executable, "train.py", "--model", config, "--weights", weights,
               "--data", str(data_yaml), "--hyp", "configs/hyp_paper.yaml",
               "--epochs", "1", "--batch", "2", "--imgsz", "64",
               "--device", device_override or (str(device) if device.type == "cuda" else "cpu"),
               "--workers", "0", "--project", str(output / "mini_runs"),
               "--name", name, "--no-val", "--no-plots", "--no-save"]
    log = output / f"mini_train_{name}.log"
    try:
        with log.open("w", encoding="utf-8") as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                           check=True, timeout=180 if device_override else 120)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"Mini training failed for {name}; log: {log}\n{tail}") from error
    return {"epochs": 1, "images": 2, "device": device_override or str(device),
            "passed": True, "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/dysample_verification")
    parser.add_argument("--mini-train", action="store_true",
                        help="Also train each variant for one epoch on two synthetic images")
    parser.add_argument("--mini-ddp", action="store_true",
                        help="Also run a one-epoch two-GPU DDP mini training when available")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    source = YOLO(args.weights).model.to(device)
    image = torch.randn(1, 3, 640, 640, device=device)
    report = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "precision": "FP32",
        "batch": 1,
        "input_shape": list(image.shape),
        "models": {},
    }
    for name, config in MODELS.items():
        result = inspect_model(name, config, source, image, output)
        if args.mini_train:
            result["mini_train"] = run_small_training(config, name, args.weights, device, output)
        report["models"][name] = result
        print(name, "parameters", result["trainable_parameters"],
              "latency_ms", round(result["inference"]["median_ms"], 2),
              "FLOPs_G", result["flops_g"], flush=True)
    if args.mini_ddp:
        if torch.cuda.device_count() >= 2:
            report["ddp_mini_train"] = run_small_training(
                MODELS["Paddy-YOLO-DySample"], "proposed_ddp", args.weights,
                device, output, device_override="0,1")
            print("Two-GPU DDP mini training passed", flush=True)
        else:
            report["ddp_mini_train"] = {"skipped": "Fewer than two CUDA GPUs available"}
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
