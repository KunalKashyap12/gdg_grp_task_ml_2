"""
Phase 10: Comprehensive REST API Integration and Unit Tests.
Tests:
- GET /
- GET /health
- GET /supported-classes
- POST /predict (valid image, calibrated confidence, top-k candidates)
- POST /predict error handling (invalid file format, corrupted bytes, empty payload)
- Mandatory agricultural disclaimer verification on all responses
"""

import io
from PIL import Image
from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.schemas import MANDATORY_DISCLAIMER


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_test_image_bytes(format="JPEG", size=(250, 250), color=(80, 160, 60)):
    """Generates synthetic image byte buffer for test uploads."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Crop Leaf Disease Classification API"
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["num_classes"] == 37
    assert data["supported_crops_count"] == 10
    assert data["calibration_active"] is True
    assert data["temperature"] > 0
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_supported_classes_endpoint(client):
    response = client.get("/supported-classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 37
    assert data["total_crops"] == 10

    expected_crops = {
        "Tomato", "Potato", "Pepper", "Maize", "Apple",
        "Grape", "Peach", "Strawberry", "Cherry", "Rice"
    }
    assert set(data["crops"]) == expected_crops
    assert len(data["classes"]) == 37

    # Ensure every crop has at least Healthy and one disease
    classes_by_crop = {}
    for c in data["classes"]:
        classes_by_crop.setdefault(c["crop"], []).append(c["disease"])

    for crop, diseases in classes_by_crop.items():
        assert "Healthy" in diseases, f"Crop {crop} missing Healthy class"
        assert len(diseases) >= 2, f"Crop {crop} must have at least one disease class"

    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_predict_endpoint_valid_image(client):
    img_bytes = create_test_image_bytes(format="JPEG")
    files = {"file": ("leaf_sample.jpg", img_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    # Verify prediction structure
    assert "prediction" in data
    pred = data["prediction"]
    assert 0 <= pred["class_id"] < 37
    assert len(pred["crop"]) > 0
    assert len(pred["disease"]) > 0
    assert 0.0 <= pred["calibrated_confidence"] <= 1.0

    # Verify candidate ranking
    assert "top_candidates" in data
    assert len(data["top_candidates"]) == 3
    assert data["top_candidates"][0]["rank"] == 1
    assert data["top_candidates"][0]["calibrated_confidence"] >= data["top_candidates"][1]["calibrated_confidence"]

    # Verify calibration metadata
    assert data["calibration"]["is_calibrated"] is True
    assert data["calibration"]["temperature"] > 0

    # Verify performance
    assert data["performance"]["latency_ms"] > 0

    # Verify mandatory disclaimer
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_predict_endpoint_custom_top_k(client):
    img_bytes = create_test_image_bytes(format="PNG")
    files = {"file": ("foliage.png", img_bytes, "image/png")}
    response = client.post("/predict?top_k=5", files=files)

    assert response.status_code == 200
    data = response.json()
    assert len(data["top_candidates"]) == 5
    for i in range(5):
        assert data["top_candidates"][i]["rank"] == i + 1


def test_predict_endpoint_invalid_file_extension(client):
    files = {"file": ("notes.txt", b"Bacterial blight symptoms", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_predict_endpoint_empty_file(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_predict_endpoint_corrupted_image(client):
    corrupt_bytes = b"GIF89a corrupted header not real image data 1234567890"
    files = {"file": ("corrupted.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "not a valid or readable image" in response.json()["detail"]
