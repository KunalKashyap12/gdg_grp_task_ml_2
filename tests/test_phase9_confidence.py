"""
Phase 9 Verification Tests:
Tests confidence calibration, low-confidence threshold optimization,
model_config.json integrity, and accuracy-coverage curve artifacts.
"""

import os
import json
import pytest


def test_model_config_json_exists_and_valid():
    config_path = "models/model_config.json"
    assert os.path.exists(config_path), f"Missing {config_path}"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    assert cfg["model_name"] == "mobilenet_v3_large"
    assert cfg["model_version"] in ["1.0.0", "1.1.0", "1.1.1"]
    assert cfg["temperature"] > 0.0
    assert 0.5 <= cfg["confidence_threshold"] <= 0.95
    assert cfg["validation_metrics"]["selective_accuracy"] >= 0.95, "Expected >=95% accuracy on accepted validation predictions"
    assert cfg["test_metrics"]["selective_accuracy"] >= 0.94, "Expected >=94% accuracy on accepted test predictions"
    assert cfg["plantdoc_metrics"]["rejected_pct"] > 50.0, "Expected >50% of out-of-domain PlantDoc images to be flagged as low-confidence"
    assert "AI-assisted indication only" in cfg["disclaimer"]


def test_accuracy_vs_coverage_plot_exists():
    plot_path = "reports/accuracy_vs_coverage.png"
    assert os.path.exists(plot_path), f"Missing {plot_path}"
    assert os.path.getsize(plot_path) > 10000, f"Plot {plot_path} is suspiciously small"
