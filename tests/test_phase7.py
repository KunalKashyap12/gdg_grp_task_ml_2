"""
Phase 7 Verification Tests:
Tests test-set evaluation artifacts, per-class metrics completeness,
confusion matrices, Grad-CAM maps, and domain shift metrics.
"""

import os
import json
import pandas as pd
import pytest


def test_per_class_metrics_csv_exists_and_complete():
    csv_path = "reports/test_metrics_per_class.csv"
    assert os.path.exists(csv_path), "reports/test_metrics_per_class.csv missing"
    df = pd.read_csv(csv_path)
    assert len(df) == 37, f"Expected 37 classes, found {len(df)}"
    for col in ["class_id", "label", "crop", "disease", "precision", "recall", "f1_score", "support"]:
        assert col in df.columns, f"Missing column {col} in test_metrics_per_class.csv"
    assert (df["precision"] >= 0.0).all() and (df["precision"] <= 1.0).all()
    assert (df["recall"] >= 0.0).all() and (df["recall"] <= 1.0).all()
    assert (df["f1_score"] >= 0.0).all() and (df["f1_score"] <= 1.0).all()
    assert df["support"].sum() == 2829, f"Expected 2829 total test samples, got {df['support'].sum()}"


def test_visual_diagnostics_artifacts_exist():
    required_plots = [
        "reports/confusion_matrix.png",
        "reports/confusion_matrix_normalized.png",
        "reports/misclassified_gallery.png",
        "reports/gradcam_visualizations.png"
    ]
    for p in required_plots:
        assert os.path.exists(p), f"Artifact missing: {p}"
        assert os.path.getsize(p) > 10000, f"File {p} is empty or suspiciously small ({os.path.getsize(p)} bytes)"


def test_plantdoc_domain_shift_metrics():
    json_path = "reports/plantdoc_domain_shift_metrics.json"
    assert os.path.exists(json_path), "reports/plantdoc_domain_shift_metrics.json missing"
    with open(json_path, "r") as f:
        data = json.load(f)
    assert data["samples"] == 3996
    assert "top1_acc" in data
    assert "top3_acc" in data
    assert "domain_shift_drop_acc" in data
    assert data["top3_acc"] > data["top1_acc"], "Top-3 accuracy should be higher than Top-1"


def test_evaluation_report_completeness():
    report_path = "reports/evaluation_report.md"
    assert os.path.exists(report_path), "reports/evaluation_report.md missing"
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Mandatory disclaimer
    assert "The output is an AI-assisted indication, NOT a guaranteed agricultural diagnosis." in text
    assert "Grad-CAM" in text
    assert "PlantDoc" in text
    assert "Crop Identification" in text
