"""
Phase 4 Verification Tests: Preprocessing, Shared Transforms, and Augmentations.
"""

import os
import io
import pytest
import numpy as np
from PIL import Image
import torch

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    preprocess_image_for_inference,
    CropLeafDataset,
    IMAGENET_MEAN,
    IMAGENET_STD
)


def test_train_transforms():
    tf = get_train_transforms(img_size=224)
    dummy_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    res = tf(image=dummy_img)["image"]
    assert isinstance(res, torch.Tensor)
    assert res.shape == (3, 224, 224)
    assert res.dtype == torch.float32


def test_val_transforms_deterministic():
    tf = get_val_transforms(img_size=224, method="center_crop")
    dummy_img = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    res1 = tf(image=dummy_img)["image"]
    res2 = tf(image=dummy_img)["image"]
    assert isinstance(res1, torch.Tensor)
    assert res1.shape == (3, 224, 224)
    # Must be 100% deterministic
    assert torch.allclose(res1, res2), "Validation transforms must have zero stochasticity"


def test_preprocess_for_inference():
    # Test with PIL Image
    pil_img = Image.new("RGB", (320, 240), color=(100, 150, 50))
    t1 = preprocess_image_for_inference(pil_img, img_size=224, return_tensor=True)
    assert isinstance(t1, torch.Tensor)
    assert t1.shape == (1, 3, 224, 224)

    # Test with bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    t2 = preprocess_image_for_inference(img_bytes, img_size=224, return_tensor=True)
    assert isinstance(t2, torch.Tensor)
    assert t2.shape == (1, 3, 224, 224)

    # Test with NumPy array
    np_arr = np.array(pil_img)
    t3 = preprocess_image_for_inference(np_arr, img_size=224, return_tensor=False)
    assert isinstance(t3, np.ndarray)
    assert t3.shape == (1, 3, 224, 224)


def test_augmentation_reports_exist():
    assert os.path.exists("reports/augmented_samples_grid.png")
    assert os.path.exists("reports/phase4_preprocessing_and_augmentation_report.md")
