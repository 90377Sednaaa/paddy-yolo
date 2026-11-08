"""Model loader and runtime manager for Paddy YOLO models.

Integrates custom Paddy-YOLO modules (RGG, RepConv, EMA, DySample) with Ultralytics YOLO
and provides robust fallback simulation when checkpoint weights are being positioned.
Automatically discovers any .pt file placed in the dedicated model folders (original vs improved).
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure repository root and backend directory are on sys.path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register custom Paddy-YOLO blocks into Ultralytics
try:
    try:
        from paddy_yolo.modules.register import register_paddy_modules
    except ImportError:
        from simulator.backend.paddy_yolo.modules.register import register_paddy_modules

    register_paddy_modules()
    PADDY_MODULES_AVAILABLE = True
except Exception as e:
    PADDY_MODULES_AVAILABLE = False
    print(f"Notice: Paddy custom modules registration deferred: {e}")

try:
    from simulator.backend.stats_data import CLASS_NAMES
except ImportError:
    from stats_data import CLASS_NAMES

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
ORIGINAL_FOLDER = WEIGHTS_DIR / "original"
IMPROVED_FOLDER = WEIGHTS_DIR / "improved"


def resolve_model_weights(folder: Path, fallback_filename: str) -> Optional[Path]:
    """Find any .pt weight file in the target folder regardless of filename.
    
    If multiple .pt files exist, prefers 'best.pt' first, otherwise the newest file.
    Falls back to checking the parent weights directory for the legacy filename.
    """
    if folder.exists() and folder.is_dir():
        pt_files = list(folder.glob("*.pt"))
        if pt_files:
            # Prioritize best.pt
            for f in pt_files:
                if f.name.lower() == "best.pt":
                    return f
            # Otherwise sort by modification time (most recent first)
            pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return pt_files[0]

    # Fallback to direct file in weights root if present
    direct_fallback = WEIGHTS_DIR / fallback_filename
    if direct_fallback.exists() and direct_fallback.is_file():
        return direct_fallback

    return None


class PaddyDetector:
    """Detection interface wrapping either Ultralytics YOLO or high-fidelity simulation."""

    def __init__(
        self,
        name: str,
        folder_path: Path,
        fallback_filename: str,
        expected_latency_range_ms: Tuple[float, float] = (14.0, 24.0),
    ):
        self.name = name
        self.folder_path = folder_path
        self.fallback_filename = fallback_filename
        self.expected_latency_range_ms = expected_latency_range_ms
        self.weights_path: Optional[Path] = None
        self.model = None
        self.is_mock = True
        self._load_model()

    def _load_model(self) -> None:
        self.weights_path = resolve_model_weights(self.folder_path, self.fallback_filename)
        if self.weights_path and self.weights_path.exists() and self.weights_path.is_file():
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(self.weights_path))
                self.is_mock = False
                print(f"Successfully loaded {self.name} weights from: {self.weights_path}")
            except Exception as e:
                print(f"Warning: Failed to load {self.weights_path}: {e}. Operating in simulation mode.")
                self.model = None
                self.is_mock = True
        else:
            self.model = None
            self.is_mock = True

    def reload(self) -> bool:
        """Attempt to discover and reload weights from disk."""
        self._load_model()
        return not self.is_mock

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run object detection on a PIL image and return detections with latency breakdown."""
        t_start = time.perf_counter()
        orig_w, orig_h = image.size

        if not self.is_mock and self.model is not None:
            # Native PyTorch / Ultralytics inference
            t_pre_start = time.perf_counter()
            img_np = np.array(image.convert("RGB"))
            preprocess_ms = (time.perf_counter() - t_pre_start) * 1000.0

            t_infer_start = time.perf_counter()
            results = self.model.predict(source=img_np, conf=0.25, verbose=False)
            inference_ms = (time.perf_counter() - t_infer_start) * 1000.0

            t_post_start = time.perf_counter()
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    species = CLASS_NAMES.get(cls_id, f"Species {cls_id}")
                    detections.append(
                        {
                            "bbox": [round(c, 2) for c in xyxy],
                            "confidence": round(conf, 4),
                            "class_id": cls_id,
                            "species_name": species,
                        }
                    )
            postprocess_ms = (time.perf_counter() - t_post_start) * 1000.0
        else:
            # Intelligent fallback simulation
            t_pre_start = time.perf_counter()
            time.sleep(random.uniform(0.0015, 0.0035))  # simulate preprocess
            preprocess_ms = (time.perf_counter() - t_pre_start) * 1000.0

            t_infer_start = time.perf_counter()
            min_l, max_l = self.expected_latency_range_ms
            simulated_sleep = random.uniform(min_l, max_l) / 1000.0
            time.sleep(simulated_sleep)
            inference_ms = (time.perf_counter() - t_infer_start) * 1000.0

            t_post_start = time.perf_counter()
            detections = self._generate_simulated_detections(orig_w, orig_h)
            postprocess_ms = (time.perf_counter() - t_post_start) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "model_name": self.name,
            "is_mock": self.is_mock,
            "weights_file": self.weights_path.name if self.weights_path else None,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": [orig_w, orig_h],
            "timing": {
                "preprocess_ms": round(preprocess_ms, 2),
                "inference_ms": round(inference_ms, 2),
                "postprocess_ms": round(postprocess_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }

    def _generate_simulated_detections(self, width: int, height: int) -> List[Dict[str, Any]]:
        """Generate representative detections of rice pest insects with realistic bounds."""
        sample_candidates = [
            (0, "rice leaf roller"),
            (1, "pink rice borer"),
            (3, "yellow rice borer"),
            (4, "rice leaf caterpillar"),
            (5, "small brown plant hopper"),
            (7, "brown plant hopper"),
            (8, "rice leaf hopper"),
            (13, "rice weevil"),
            (14, "rice shell pest"),
        ]

        # Improved model tends to identify more true positives (higher recall)
        num_detections = random.choice([1, 2, 3]) if "EMA" in self.name else random.choice([1, 2])
        selected = random.sample(sample_candidates, min(num_detections, len(sample_candidates)))

        detections = []
        for cls_id, species in selected:
            w_box = random.uniform(width * 0.12, width * 0.35)
            h_box = random.uniform(height * 0.12, height * 0.35)
            x1 = random.uniform(width * 0.08, width - w_box - width * 0.08)
            y1 = random.uniform(height * 0.08, height - h_box - height * 0.08)
            x2 = x1 + w_box
            y2 = y1 + h_box

            base_conf = random.uniform(0.72, 0.94) if "EMA" in self.name else random.uniform(0.64, 0.88)

            detections.append(
                {
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(base_conf, 4),
                    "class_id": cls_id,
                    "species_name": species,
                }
            )

        return detections


# Singleton detector instances
baseline_detector = PaddyDetector(
    name="Paddy YOLO",
    folder_path=ORIGINAL_FOLDER,
    fallback_filename="paddy_yolo_baseline.pt",
    expected_latency_range_ms=(16.0, 21.0),
)

improved_detector = PaddyDetector(
    name="Paddy YOLO EMA + Dysample",
    folder_path=IMPROVED_FOLDER,
    fallback_filename="paddy_yolo_ema_dysample.pt",
    expected_latency_range_ms=(12.5, 16.5),
)


def get_detectors() -> Tuple[PaddyDetector, PaddyDetector]:
    """Return both model detector instances."""
    return baseline_detector, improved_detector
