"""Inference orchestrator for dual-model asynchronous execution."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from PIL import Image

from simulator.backend.model_loader import get_detectors

_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_dual_inference(image: Image.Image) -> Dict[str, Any]:
    """Execute both models concurrently in separate worker threads to benchmark asynchronous speed."""
    baseline_detector, improved_detector = get_detectors()

    loop = asyncio.get_running_loop()

    # Execute predictions in thread pool concurrently
    task_baseline = loop.run_in_executor(_thread_pool, baseline_detector.predict, image)
    task_improved = loop.run_in_executor(_thread_pool, improved_detector.predict, image)

    baseline_result, improved_result = await asyncio.gather(task_baseline, task_improved)

    t_base = baseline_result["timing"]["total_ms"]
    t_impr = improved_result["timing"]["total_ms"]

    delta_ms = abs(t_base - t_impr)
    if t_impr < t_base:
        winner = "Paddy YOLO EMA + Dysample"
        speedup_pct = round(((t_base - t_impr) / t_base) * 100.0, 1) if t_base > 0 else 0.0
    elif t_base < t_impr:
        winner = "Paddy YOLO"
        speedup_pct = round(((t_impr - t_base) / t_impr) * 100.0, 1) if t_impr > 0 else 0.0
    else:
        winner = "Tie"
        speedup_pct = 0.0

    return {
        "baseline": baseline_result,
        "improved": improved_result,
        "showdown": {
            "winner": winner,
            "faster_model": winner,
            "delta_ms": round(delta_ms, 2),
            "speedup_pct": speedup_pct,
            "baseline_ms": t_base,
            "improved_ms": t_impr,
            "baseline_fps": round(1000.0 / t_base, 1) if t_base > 0 else 0.0,
            "improved_fps": round(1000.0 / t_impr, 1) if t_impr > 0 else 0.0,
        },
    }
