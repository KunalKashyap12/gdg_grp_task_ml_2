"""
Phase 9 Verification Tests:
Tests ONNX model integrity, dynamic batching, numeric parity with PyTorch,
and inference engine functionality with calibrated confidence.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import onnx
import onnxruntime as ort
import pytest

from src.serve.inference import get_inference_engine, MANDATORY_DISCLAIMER
from src.train.models import build_model


def test_onnx_model_file_and_validity():
    onnx_path = "models/crop_disease_model.onnx"
    assert os.path.exists(onnx_path), "models/crop_disease_model.onnx missing"
    assert os.path.getsize(onnx_path) > 10 * 1024 * 1024, "ONNX model unexpectedly small"
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)


def test_numerical_parity_pytorch_vs_onnx():
    # Load PyTorch model
    ckpt = torch.load("models/best_model.pt", map_location="cpu")
    pt_model = build_model("mobilenet_v3_large", num_classes=37, pretrained=False)
    pt_model.load_state_dict(ckpt["state_dict"])
    pt_model.eval()

    # Load ONNX session
    ort_session = ort.InferenceSession("models/crop_disease_model.onnx", providers=["CPUExecutionProvider"])

    for bs in [1, 3]:
        dummy = torch.randn(bs, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_model(dummy).numpy()

        ort_out = ort_session.run(None, {"input": dummy.numpy()})[0]
        assert np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-3), f"Parity mismatch for batch {bs}"


def test_inference_engine_prediction_schema():
    engine = get_inference_engine()
    test_img = Image.new("RGB", (250, 250), color=(100, 180, 50))
    res = engine.predict(test_img, top_k=3)

    assert "prediction" in res
    assert "class_id" in res["prediction"]
    assert "crop" in res["prediction"]
    assert "disease" in res["prediction"]
    assert "calibrated_confidence" in res["prediction"]
    assert 0.0 <= res["prediction"]["calibrated_confidence"] <= 1.0

    assert "top_candidates" in res
    assert len(res["top_candidates"]) == 3
    assert res["top_candidates"][0]["rank"] == 1

    assert "calibration" in res
    assert res["calibration"]["temperature"] > 0

    assert "performance" in res
    assert res["performance"]["latency_ms"] < 200.0

    assert res["disclaimer"] == MANDATORY_DISCLAIMER
