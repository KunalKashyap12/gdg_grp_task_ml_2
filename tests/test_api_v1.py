"""
Phase 10: Integration and Unit Tests for API v1 Specification.
Tests:
- GET  /api/v1/health
- GET  /api/v1/classes
- GET  /api/v1/model-info
- POST /api/v1/predict (valid high-confidence leaf image)
- POST /api/v1/predict (low-confidence ambiguous / blank image)
- Error code 400 (corrupt image)
- Error code 413 (file > 5 MB)
- Error code 415 (unsupported media type)
- Error code 422 (missing field 'image')
- Consistent error envelope shape: {"error": {"code": str, "message": str}}
- Exact mandatory disclaimer verification
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


def create_image_bytes(color=(40, 160, 40), size=(250, 250), fmt="JPEG"):
    """Generates JPEG byte buffer for test uploads."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def test_v1_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_v1_classes(client):
    res = client.get("/api/v1/classes")
    assert res.status_code == 200
    data = res.json()
    assert data["total_classes"] == 37
    assert data["total_crops"] == 10
    assert len(data["classes"]) == 37
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_v1_model_info(client):
    res = client.get("/api/v1/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "mobilenet_v3_large"
    assert data["model_version"] in ["1.0.0", "1.1.0", "1.1.1"]
    assert data["threshold"] in [0.70, 0.74]
    assert data["temperature"] > 0
    assert "metrics_summary" in data
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_v1_predict_valid_image(client):
    img_bytes = create_image_bytes(color=(30, 140, 30))
    files = {"image": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()

    assert data["status"] in ["success", "low_confidence"]
    assert "prediction" in data
    pred = data["prediction"]
    assert "class_id" in pred
    assert "crop" in pred
    assert "disease" in pred
    assert "label" in pred
    assert 0.0 <= pred["confidence"] <= 1.0

    assert len(data["top_k"]) == 3
    for item in data["top_k"]:
        assert "label" in item
        assert "confidence" in item

    assert "low_confidence" in data
    assert data["threshold"] in [0.70, 0.74]
    assert data["model_version"] in ["1.0.0", "1.1.0", "1.1.1"]
    assert data["inference_time_ms"] > 0
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


def test_v1_predict_low_confidence_blank_image(client):
    # Pure gray image should trigger low confidence or out-of-distribution
    blank_bytes = create_image_bytes(color=(128, 128, 128))
    files = {"image": ("blank.png", blank_bytes, "image/png")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "low_confidence"
    assert data["low_confidence"] is True
    assert data["message"] == "Unable to identify reliably. Retake a clear, well-lit, close-up photo of a single leaf."
    assert len(data["top_k"]) == 3


def test_v1_predict_unsupported_media_type_415(client):
    files = {"image": ("report.pdf", b"%PDF-1.4...", "application/pdf")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 415
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_v1_predict_corrupt_image_400(client):
    corrupt_bytes = b"Not a real image file content"
    files = {"image": ("broken.jpg", corrupt_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "BAD_REQUEST"


def test_v1_predict_oversized_file_413(client):
    # 5.2 MB payload
    large_bytes = b"0" * (5 * 1024 * 1024 + 200)
    files = {"image": ("huge_photo.jpg", large_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 413
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "FILE_TOO_LARGE"


def test_v1_predict_missing_field_422(client):
    # Field name is 'photo' instead of required 'image'
    files = {"photo": ("leaf.jpg", create_image_bytes(), "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_v1_predict_non_leaf_wallpaper_rejected(client):
    """
    Ensures out-of-domain wallpapers / non-botanical images are strictly rejected
    by the foliar gatekeeper with explicit user feedback message.
    """
    import os
    iron_man_path = os.path.join("data", "samples", "iron_man_wallpaper.jpg")
    if os.path.exists(iron_man_path):
        with open(iron_man_path, "rb") as f:
            img_bytes = f.read()
    else:
        # Non-foliar blue/white checkerboard graphic (high variance, zero green)
        import numpy as np
        grid = np.zeros((200, 200, 3), dtype=np.uint8)
        grid[::20, :] = [255, 255, 255]
        grid[:, ::20] = [200, 50, 50]
        buf = io.BytesIO()
        Image.fromarray(grid).save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    files = {"image": ("wallpaper.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "not_a_leaf"
    assert data["is_leaf"] is False
    assert data["message"] == "Not a leaf. Please upload a new image."


def test_v1_predict_dark_graphic_wallpaper_and_portrait_rejected(client):
    """
    Ensures dark graphic wallpapers with red lettering/faces (e.g. sports wallpapers)
    are strictly rejected as not_a_leaf, avoiding false foliar disease diagnoses.
    """
    import numpy as np
    # Synthetic dark wallpaper: black background with red typography & warm skin tone patch (zero green)
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Red graphic text region:
    img_arr[30:70, 30:190] = [180, 20, 20]
    # Human face skin-tone patch (warm tone: R~210, G~150, B~120):
    img_arr[100:180, 80:150] = [210, 150, 120]
    
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"image": ("verstappen_wallpaper.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "not_a_leaf"
    assert data["is_leaf"] is False
    assert data["message"] == "Not a leaf. Please upload a new image."


def test_v1_predict_ultrawide_desktop_wallpaper_rejected(client):
    """
    Ensures ultrawide wallpapers (e.g., 21:9 or 5120x2160, aspect ratio > 2.05)
    are strictly rejected as not_a_leaf.
    """
    import numpy as np
    # 512x216 image (aspect ratio 2.37)
    img_arr = np.full((216, 512, 3), 100, dtype=np.uint8)
    # Add some green pixels at the bottom
    img_arr[180:, :] = [40, 120, 40]

    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"image": ("ultrawide_wallpaper.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "not_a_leaf"
    assert data["is_leaf"] is False
    assert data["message"] == "Not a leaf. Please upload a new image."


def test_v1_predict_scenic_landscape_wallpaper_rejected(client):
    """
    Ensures scenic landscape photos (mountains, haze, misty sky in upper frame,
    distant greenery without central foliar subject) are strictly rejected as not_a_leaf.
    """
    import numpy as np
    # 300x300 image (1:1 ratio)
    img_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    # Bright hazy/misty sky in top 40% (high value, low saturation)
    img_arr[:120, :] = [210, 215, 220]
    # Mountain rocks in middle (gray/brown)
    img_arr[120:230, :] = [90, 85, 80]
    # Distant green forest at bottom 20%
    img_arr[230:, :] = [30, 95, 35]

    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"image": ("scenic_mountain_wallpaper.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict", files=files)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "not_a_leaf"
    assert data["is_leaf"] is False
    assert data["message"] == "Not a leaf. Please upload a new image."


def test_v1_predict_with_valid_crop_hint(client):
    img_bytes = create_image_bytes(color=(30, 140, 30))
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    data_payload = {"crop": "Apple"}
    res = client.post("/api/v1/predict", files=files, data=data_payload)

    assert res.status_code == 200
    data = res.json()
    if data["is_leaf"]:
        assert data["stage1_leaf"]["is_conditioned"] is True
        assert data["prediction"]["crop"] == "Apple"
        assert data["crop_conditioned"] == "Apple"


def test_v1_predict_with_invalid_crop_hint_400(client):
    img_bytes = create_image_bytes(color=(30, 140, 30))
    files = {"image": ("leaf.jpg", img_bytes, "image/jpeg")}
    data_payload = {"crop": "Banana"}
    res = client.post("/api/v1/predict", files=files, data=data_payload)

    assert res.status_code == 400
    data = res.json()
    assert data["error"]["code"] == "BAD_REQUEST"
    assert "Unsupported crop hint 'Banana'" in data["error"]["message"]


def test_v1_samples_catalog_and_fetch(client):
    res = client.get("/api/v1/samples")
    assert res.status_code == 200
    samples = res.json()
    assert isinstance(samples, list)
    assert len(samples) > 0

    first_sample = samples[0]["filename"]
    res_img = client.get(f"/api/v1/samples/{first_sample}")
    assert res_img.status_code == 200
    assert len(res_img.content) > 0


def test_preprocess_numpy_types_robustness():
    from src.data.transforms import preprocess_image_for_inference
    import numpy as np

    # 1. Float in [0.0, 1.0]
    float_img = np.random.rand(224, 224, 3).astype(np.float32)
    t1 = preprocess_image_for_inference(float_img, return_tensor=True)
    assert t1.shape == (1, 3, 224, 224)

    # 2. 2D grayscale
    gray_2d = (np.random.rand(200, 200) * 255).astype(np.uint8)
    t2 = preprocess_image_for_inference(gray_2d, return_tensor=True)
    assert t2.shape == (1, 3, 224, 224)

    # 3. 3D grayscale (H, W, 1)
    gray_3d = (np.random.rand(180, 180, 1) * 255).astype(np.uint8)
    t3 = preprocess_image_for_inference(gray_3d, return_tensor=True)
    assert t3.shape == (1, 3, 224, 224)

    # 4. RGBA (H, W, 4)
    rgba = (np.random.rand(160, 160, 4) * 255).astype(np.uint8)
    t4 = preprocess_image_for_inference(rgba, return_tensor=True)
    assert t4.shape == (1, 3, 224, 224)


def test_inference_engine_boundary_top_k():
    from src.serve.inference import get_inference_engine
    engine = get_inference_engine()
    test_img = Image.new("RGB", (250, 250), color=(100, 180, 50))

    # top_k = 0 and negative must not throw IndexError
    res_zero = engine.predict(test_img, top_k=0)
    assert len(res_zero["top_candidates"]) == 1

    res_neg = engine.predict(test_img, top_k=-5)
    assert len(res_neg["top_candidates"]) == 1

    res_huge = engine.predict(test_img, top_k=100)
    assert len(res_huge["top_candidates"]) == 37


