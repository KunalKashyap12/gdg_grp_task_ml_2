"""
Phase 8 Verification Tests:
Tests robustness outputs: PlantDoc domain shift drop, synthetic stress test logs,
out-of-scope evaluation metrics, and comprehensive robustness report.
"""

import os
import json
import pandas as pd
import pytest


def test_robustness_stress_tests_csv():
    path = "reports/robustness_stress_tests.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 13, f"Expected 13 conditions (1 baseline + 12 stress), got {len(df)}"
    for col in ["condition", "top1_accuracy", "top3_accuracy", "macro_f1", "accuracy_delta_pct"]:
        assert col in df.columns, f"Missing column {col}"
    assert df.loc[df["condition"] == "baseline", "accuracy_delta_pct"].values[0] == 0.0


def test_out_of_scope_evaluation_json():
    path = "reports/out_of_scope_evaluation.json"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r") as f:
        data = json.load(f)
    assert data["total_tested"] >= 30
    assert "mean_confidence" in data
    assert "mean_entropy" in data
    assert "records" in data


def test_robustness_report_markdown():
    path = "reports/robustness_report.md"
    assert os.path.exists(path), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    assert "PlantDoc" in text
    assert "Known Weaknesses" in text
    assert "AI-assisted indication only" in text
