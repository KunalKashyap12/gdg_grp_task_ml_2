"""
Phase 8 Verification Tests:
Tests post-hoc temperature scaling calibration, ECE reduction,
reliability curves, and strictly positive temperature parameterization.
"""

import os
import json
import torch
import pytest
from src.calibrate.temperature_scaling import ModelWithTemperature


def test_calibration_json_exists_and_valid():
    cal_path = "models/calibration.json"
    assert os.path.exists(cal_path), "models/calibration.json missing"
    with open(cal_path, "r") as f:
        data = json.load(f)

    assert "temperature" in data
    assert data["temperature"] > 0.0, "Temperature must be strictly positive"
    assert data["method"] == "temperature_scaling"

    # Validation ECE reduction
    val_pre = data["validation"]["uncalibrated_ece"]
    val_post = data["validation"]["calibrated_ece"]
    assert val_post < val_pre, f"Calibrated ECE ({val_post}) should be lower than uncalibrated ({val_pre})"
    assert data["validation"]["relative_reduction_pct"] > 50.0, "Expected >50% relative ECE reduction"

    # Test ECE reduction
    test_pre = data["test"]["uncalibrated_ece"]
    test_post = data["test"]["calibrated_ece"]
    assert test_post < test_pre, f"Test Calibrated ECE ({test_post}) should be lower than uncalibrated ({test_pre})"
    assert data["test"]["relative_reduction_pct"] > 50.0, "Expected >50% test relative ECE reduction"

    # Disclaimer
    assert data["disclaimer"] == "The output is an AI-assisted indication, NOT a guaranteed agricultural diagnosis."


def test_calibration_curve_plot_exists():
    plot_path = "reports/calibration_curves.png"
    assert os.path.exists(plot_path), "reports/calibration_curves.png missing"
    assert os.path.getsize(plot_path) > 10000, f"File {plot_path} is suspiciously small ({os.path.getsize(plot_path)} bytes)"


def test_temperature_scaler_strictly_positive_and_order_preserving():
    dummy_model = torch.nn.Linear(10, 5)
    scaler = ModelWithTemperature(dummy_model)
    assert scaler.temperature > 0.0

    dummy_input = torch.randn(4, 10)
    unscaled_logits = dummy_model(dummy_input)
    scaled_logits = scaler(dummy_input)

    # Temperature scaling preserves ranking of classes
    assert torch.equal(unscaled_logits.argmax(dim=1), scaled_logits.argmax(dim=1))

    # Softmax probabilities sum to 1
    probs = torch.softmax(scaled_logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
