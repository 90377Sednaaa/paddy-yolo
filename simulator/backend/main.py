"""FastAPI application for Paddy YOLO Dual-Model Comparison Simulator."""

import io
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from simulator.backend.inference import run_dual_inference
from simulator.backend.model_loader import (
    BASELINE_WEIGHTS,
    IMPROVED_WEIGHTS,
    PADDY_MODULES_AVAILABLE,
    get_detectors,
)
from simulator.backend.stats_data import ARCHITECTURE_SPECS, MACRO_METRICS, SPECIES_DATA

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

app = FastAPI(
    title="Paddy YOLO Model Comparison Simulator API",
    description="Asynchronous dual-model computer vision benchmark API comparing Paddy YOLO against Paddy YOLO EMA + Dysample.",
    version="1.0.0",
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Health check and model weight availability report."""
    base_det, impr_det = get_detectors()
    return {
        "status": "healthy",
        "custom_modules_registered": PADDY_MODULES_AVAILABLE,
        "models": {
            "baseline": {
                "name": base_det.name,
                "weights_found": BASELINE_WEIGHTS.exists(),
                "weights_path": str(BASELINE_WEIGHTS),
                "is_mock": base_det.is_mock,
            },
            "improved": {
                "name": impr_det.name,
                "weights_found": IMPROVED_WEIGHTS.exists(),
                "weights_path": str(IMPROVED_WEIGHTS),
                "is_mock": impr_det.is_mock,
            },
        },
    }


@app.get("/api/statistics")
def get_statistics() -> Dict[str, Any]:
    """Return clean research evaluation benchmarks without internal hyperparameter codes."""
    return {
        "macro": MACRO_METRICS,
        "species": SPECIES_DATA,
        "architecture": ARCHITECTURE_SPECS,
    }


@app.get("/api/samples")
def get_sample_images() -> Dict[str, Any]:
    """List sample test images available for instant evaluation."""
    items = [
        {
            "id": "sample_rice_leaf_roller.jpg",
            "name": "Rice Leaf Roller",
            "description": "Foliage specimen (Cnaphalocrocis medinalis)",
            "url": "/api/samples/sample_rice_leaf_roller.jpg",
        },
        {
            "id": "sample_pink_rice_borer.jpg",
            "name": "Pink Rice Borer",
            "description": "Stem base larva (Sesamia inferens)",
            "url": "/api/samples/sample_pink_rice_borer.jpg",
        },
        {
            "id": "sample_brown_plant_hopper.jpg",
            "name": "Brown Plant Hopper",
            "description": "Tiller cluster (Nilaparvata lugens)",
            "url": "/api/samples/sample_brown_plant_hopper.jpg",
        },
        {
            "id": "sample_yellow_rice_borer.jpg",
            "name": "Yellow Rice Borer",
            "description": "Flag leaf pest (Scirpophaga incertulas)",
            "url": "/api/samples/sample_yellow_rice_borer.jpg",
        },
    ]
    return {"samples": items}


@app.get("/api/samples/{filename}")
def serve_sample(filename: str):
    """Serve a sample image file."""
    file_path = SAMPLES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Receive an uploaded insect image and run both models asynchronously."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    results = await run_dual_inference(image)
    return results
