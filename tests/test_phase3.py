"""
Phase 3 Verification Tests: Class Balance & Weighting Strategy.
"""

import os
import json
import torch
import pytest
import pandas as pd


def test_class_weights_json():
    json_path = "data/class_weights.json"
    assert os.path.exists(json_path), f"{json_path} missing"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["n_classes"] == 37
    assert len(data["standard_weights"]) == 37
    assert len(data["smoothed_weights"]) == 37
    assert len(data["weight_tensor_order"]) == 37
    
    # Check that minority classes have higher weights than majority classes
    std_w = data["standard_weights"]
    assert std_w["Rice___Leaf_blast"] > std_w["Tomato___Late_blight"], "Minority class should have higher weight"
    assert std_w["Potato___Healthy"] > std_w["Apple___Apple_scab"], "Minority class should have higher weight"


def test_class_weights_pt():
    pt_path = "models/class_weights.pt"
    assert os.path.exists(pt_path), f"{pt_path} missing"
    t = torch.load(pt_path)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (37,)
    assert torch.all(t > 0), "All weights must be positive"
    assert torch.all(torch.isfinite(t)), "All weights must be finite"


def test_reports_and_plots():
    assert os.path.exists("reports/class_counts_before_after.png"), "Comparison plot missing"
    assert os.path.exists("reports/phase3_class_balance_report.md"), "Report missing"
