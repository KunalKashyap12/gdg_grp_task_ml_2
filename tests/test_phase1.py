"""
Phase 1 Verification Tests:
1. Verify config.yaml supports all 10 crops modularly.
2. Verify supported_classes.json schema (class_id, crop, disease, display_name, description).
3. Verify every crop has a Healthy class and at least one disease class.
4. Verify all classes have >= 100 images after cleaning.
5. Verify partition integrity and zero data leakage across train/val/test.
"""

import os
import json
import yaml
import pytest
import pandas as pd


def test_config_crops():
    assert os.path.exists("config.yaml"), "config.yaml not found"
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    crops = cfg.get("crops", {})
    expected_crops = [
        "Tomato", "Potato", "Pepper", "Maize", "Apple",
        "Grape", "Peach", "Strawberry", "Cherry", "Rice"
    ]
    assert len(crops) == 10, f"Expected 10 crops, found {len(crops)}"
    for c in expected_crops:
        assert c in crops, f"Missing crop: {c}"


def test_supported_classes_json():
    json_path = "data/supported_classes.json"
    assert os.path.exists(json_path), f"{json_path} does not exist"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    assert len(data) == 37, f"Expected 37 classes, got {len(data)}"
    
    seen_ids = set()
    seen_labels = set()
    crop_classes = {}
    
    for item in data:
        # Check required fields
        for field in ["class_id", "crop", "disease", "display_name", "description", "label"]:
            assert field in item, f"Missing field '{field}' in {item}"
            assert item[field] != "" or item[field] is not None
            
        assert item["class_id"] not in seen_ids, f"Duplicate class_id: {item['class_id']}"
        assert item["label"] not in seen_labels, f"Duplicate label: {item['label']}"
        seen_ids.add(item["class_id"])
        seen_labels.add(item["label"])
        
        # Check naming convention Crop___Disease
        assert item["label"] == f"{item['crop']}___{item['disease']}"
        
        crop_classes.setdefault(item["crop"], []).append(item["disease"])
        
    # Check that every crop has Healthy and >= 1 disease
    assert len(crop_classes) == 10
    for crop, diseases in crop_classes.items():
        has_healthy = any("healthy" in d.lower() for d in diseases)
        disease_count = sum(1 for d in diseases if "healthy" not in d.lower())
        assert has_healthy, f"Crop {crop} missing Healthy class"
        assert disease_count >= 1, f"Crop {crop} missing disease classes"


def test_dataset_audit_and_counts():
    audit_path = "reports/phase1_dataset_audit.csv"
    assert os.path.exists(audit_path), f"Audit report {audit_path} missing"
    
    df = pd.read_csv(audit_path)
    assert len(df) == 37, f"Expected 37 classes in audit, got {len(df)}"
    
    # Verify every class has at least 100 clean images
    low_count = df[df["clean_count"] < 100]
    assert len(low_count) == 0, f"Found classes with < 100 clean images: {low_count['label'].tolist()}"
    
    # Check that all classes are marked 'reliable'
    unreliable = df[df["status"] != "reliable"]
    assert len(unreliable) == 0, f"Unreliable classes: {unreliable['label'].tolist()}"


def test_manifest_partition_integrity():
    for split in ["train", "val", "test"]:
        p = f"data/manifests/{split}_manifest.csv"
        assert os.path.exists(p), f"{p} missing"
        df = pd.read_csv(p)
        assert len(df) > 0
        assert df["label"].nunique() == 37, f"Split {split} does not contain all 37 classes"
        
    # Check zero data leakage across splits
    train_df = pd.read_csv("data/manifests/train_manifest.csv")
    val_df = pd.read_csv("data/manifests/val_manifest.csv")
    test_df = pd.read_csv("data/manifests/test_manifest.csv")
    
    train_paths = set(train_df["raw_path"])
    val_paths = set(val_df["raw_path"])
    test_paths = set(test_df["raw_path"])
    
    assert len(train_paths.intersection(val_paths)) == 0, "Data leakage between train and val!"
    assert len(train_paths.intersection(test_paths)) == 0, "Data leakage between train and test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Data leakage between val and test!"


def test_processed_image_folders():
    for split in ["train", "val", "test"]:
        split_dir = os.path.join("data/processed", split)
        assert os.path.exists(split_dir), f"{split_dir} does not exist"
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        assert len(classes) == 37, f"Expected 37 class folders in {split_dir}, found {len(classes)}"
