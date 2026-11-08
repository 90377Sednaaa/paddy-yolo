"""Tests for Paddy YOLO Comparison Simulator backend API."""

import io
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
ROOT_DIR = BACKEND_DIR.parent.parent

for p in (ROOT_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient
from PIL import Image

try:
    from simulator.backend.main import app
except ImportError:
    from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data
    assert "baseline" in data["models"]
    assert "improved" in data["models"]
    assert data["models"]["baseline"]["name"] == "Paddy YOLO"
    assert data["models"]["improved"]["name"] == "Paddy YOLO EMA + Dysample"


def test_statistics_endpoint():
    response = client.get("/api/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "macro" in data
    assert "species" in data
    assert "architecture" in data
    assert len(data["species"]) == 16
    species_names = [s["species"] for s in data["species"]]
    assert "rice leaf roller" in species_names
    assert "pink rice borer" in species_names
    assert "brown plant hopper" in species_names


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

    first_sample_id = data["samples"][0]["id"]
    file_resp = client.get(f"/api/samples/{first_sample_id}")
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"] in ["image/jpeg", "image/jpg"]


def test_predict_endpoint():
    img = Image.new("RGB", (320, 320), color=(50, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test_pest.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "improved" in data
    assert "showdown" in data

    # Check timing metrics
    assert data["baseline"]["timing"]["total_ms"] > 0
    assert data["improved"]["timing"]["total_ms"] > 0
    assert "winner" in data["showdown"]
    assert data["showdown"]["winner"] in [
        "Paddy YOLO",
        "Paddy YOLO EMA + Dysample",
        "Tie",
    ]
