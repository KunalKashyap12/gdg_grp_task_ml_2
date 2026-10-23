"""
Phase 5 Verification Tests: Zero Data Leakage at File-Hash and Perceptual-Group Levels.
"""

import os
import pytest
import pandas as pd


def test_manifest_files_exist():
    for name in ["train.csv", "val.csv", "test.csv", "master_group_split.csv"]:
        path = os.path.join("data", "manifests", name)
        assert os.path.exists(path), f"Manifest {path} missing"


def test_zero_hash_leakage():
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    train_md5 = set(train_df["md5"])
    val_md5 = set(val_df["md5"])
    test_md5 = set(test_df["md5"])

    # 1. No exact MD5 hash appears in more than one split
    assert len(train_md5.intersection(val_md5)) == 0, "MD5 hash leakage between train and val!"
    assert len(train_md5.intersection(test_md5)) == 0, "MD5 hash leakage between train and test!"
    assert len(val_md5.intersection(test_md5)) == 0, "MD5 hash leakage between val and test!"


def test_zero_group_leakage():
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    # 2. No perceptual-hash group appears in more than one split
    assert len(train_groups.intersection(val_groups)) == 0, "Perceptual group leakage between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Perceptual group leakage between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Perceptual group leakage between val and test!"


def test_stratification_and_class_representation():
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")

    # Every split must represent all 37 classes
    assert train_df["class_label"].nunique() == 37, f"Train split missing classes: {train_df['class_label'].nunique()}/37"
    assert val_df["class_label"].nunique() == 37, f"Val split missing classes: {val_df['class_label'].nunique()}/37"
    assert test_df["class_label"].nunique() == 37, f"Test split missing classes: {test_df['class_label'].nunique()}/37"

    total = len(train_df) + len(val_df) + len(test_df)
    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total

    # Check approximate 70/15/15 target (+- 2% tolerance)
    assert 0.68 <= train_ratio <= 0.72, f"Train ratio out of bounds: {train_ratio:.3f}"
    assert 0.13 <= val_ratio <= 0.17, f"Val ratio out of bounds: {val_ratio:.3f}"
    assert 0.13 <= test_ratio <= 0.17, f"Test ratio out of bounds: {test_ratio:.3f}"
