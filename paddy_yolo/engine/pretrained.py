"""Transfer YOLOv8 tensors by corresponding layer, including shifted EMA models."""
from pathlib import Path

from ultralytics import YOLO

# Source indices are stock YOLOv8-N; target indices follow each architecture YAML.
EMA_DYSAMPLE_LAYER_MAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 9, 8: 10, 9: 12,
    12: 15, 15: 18, 16: 19, 18: 21, 19: 22, 21: 24, 22: 25,
}


def transfer_yolov8_weights(target, weights: str, architecture: str) -> dict:
    """Load only compatible tensors at matching topological positions.

    New EMA and DySample parameters retain their own initialization. Classifier
    tensors with COCO's incompatible class count are skipped. No index-based load
    is performed before mapping the shifted architecture.
    """
    source = YOLO(weights).model
    architecture_name = Path(architecture).stem
    if architecture_name == "paddy-yolo-ema-dysample":
        mapping = EMA_DYSAMPLE_LAYER_MAP
    else:
        # Retain existing baseline and legacy enhanced-model transfer behavior.
        mapping = {i: i for i in range(len(source.model))}
        mapping.update({
            "paddy-yolo-ema": {16: 17, 18: 19, 19: 21, 21: 23},
            "paddy-yolo-p2-ema": {16: 20, 18: 22, 19: 24, 21: 26},
        }.get(architecture_name, {}))
    target_state = target.state_dict()
    transfers = {}
    counts = {}
    for key, value in source.state_dict().items():
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "model":
            continue
        source_layer = int(parts[1])
        if source_layer not in mapping:
            continue
        target_layer = mapping[source_layer]
        mapped_key = f"model.{target_layer}." + ".".join(parts[2:])
        if mapped_key in target_state and target_state[mapped_key].shape == value.shape:
            transfers[mapped_key] = value.detach().to(dtype=target_state[mapped_key].dtype)
            counts[str(target_layer)] = counts.get(str(target_layer), 0) + 1
    if not transfers:
        raise ValueError(f"No compatible tensors found in {weights} for {architecture}")
    target.load_state_dict(transfers, strict=False)
    report = {"weights": str(weights), "architecture": str(architecture),
              "transferred_tensors": len(transfers), "target_tensors": len(target_state),
              "per_layer": counts, "layer_map": mapping}
    print(f"Transferred {len(transfers)}/{len(target_state)} compatible tensors from {weights}.")
    return report
