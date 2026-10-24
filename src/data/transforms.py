"""
Canonical Preprocessing and Augmentation Module.
Shared identically across:
- Model Training (PyTorch)
- Offline Evaluation & Metrics Calculation
- Live REST API Inference (FastAPI & ONNX Runtime)
Guarantees ZERO train/serve skew.
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import io
import numpy as np
from PIL import Image, ImageOps
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Standard ImageNet normalization parameters
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_image_path(path: str) -> str:
    """
    Robustly resolves an image path whether it is relative, absolute,
    or contains legacy/other-machine workspace roots.
    """
    if not path or not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"Invalid path type: {type(path)}")

    path_str = str(path)
    if os.path.exists(path_str):
        return path_str

    # Normalize slashes
    norm = path_str.replace("\\", "/")

    # Check relative to current working directory
    rel_norm = os.path.normpath(path_str)
    if os.path.exists(rel_norm):
        return rel_norm

    # Check for 'data/' prefix
    for marker in ["data/processed", "data/plantdoc_field_test", "data/samples", "data/raw", "data/"]:
        if marker in norm:
            idx = norm.find(marker)
            candidate = norm[idx:]
            candidate_norm = os.path.normpath(candidate)
            if os.path.exists(candidate_norm):
                return candidate_norm

    raise FileNotFoundError(f"Image not found at path: {path}")


def get_train_transforms(img_size=224):
    """
    Rich domain-robust augmentation pipeline using Albumentations:
    - RandomResizedCrop & aspect-ratio exploration
    - Flips and rotations
    - Color perturbations (brightness, contrast, gamma, hue, saturation)
    - Blur & noise (Gaussian, motion, ISO/sensor noise)
    - JPEG compression artifacts
    - Random shadows & background dropout/cutout to break lab solid-background reliance
    """
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-25, 25), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5))
        ], p=0.3),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.3),
        A.ImageCompression(quality_range=(65, 95), p=0.3),
        A.RandomShadow(p=0.3),
        # Background and foreground perturbation to prevent background over-fitting
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 36),
            hole_width_range=(16, 36),
            fill=0,
            p=0.4
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transforms(img_size=224, method="center_crop"):
    """
    Strict validation/test transform:
    - Deterministic resize + aspect-ratio preservation
    - Normalization with ImageNet mean and std
    - Zero stochastic perturbation
    """
    if method == "pad":
        # Preserve aspect ratio by scaling longest edge and padding
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])
    else:
        # Standard benchmark aspect-ratio preserving center-crop resize
        return A.Compose([
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])


class CropLeafDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper integrating Albumentations transforms.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if "filepath" in self.df.columns:
            self.paths = self.df["filepath"].values
        elif "raw_path" in self.df.columns:
            self.paths = self.df["raw_path"].values
        else:
            self.paths = self.df["path"].values
        self.labels = self.df["class_id"].values if "class_id" in self.df.columns else self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = resolve_image_path(self.paths[idx])
        target = self.labels[idx]

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            image_np = np.array(img.convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = ToTensorV2()(image=image_np)["image"]

        return image_tensor, target


def preprocess_image_for_inference(image_input, img_size=224, return_tensor=True, method="center_crop", use_tta=False):
    """
    Shared preprocessing function used by API and inference pipelines.
    Accepts:
    - bytes (e.g. from FastAPI UploadFile)
    - PIL.Image
    - NumPy array (H, W, 3), (H, W), or (H, W, 1/4) in uint8 or float
    Returns:
    - Tensor of shape (1, 3, 224, 224) or (2, 3, 224, 224) if use_tta
    """
    if isinstance(image_input, (str, os.PathLike)):
        with Image.open(image_input) as raw_img:
            raw_img = ImageOps.exif_transpose(raw_img)
            if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
                bg = Image.new("RGB", raw_img.size, (128, 128, 128))
                rgba = raw_img if raw_img.mode == "RGBA" else raw_img.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[3])
                image_np = np.array(bg)
            else:
                image_np = np.array(raw_img.convert("RGB"))
    elif isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as raw_img:
            raw_img = ImageOps.exif_transpose(raw_img)
            if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
                bg = Image.new("RGB", raw_img.size, (128, 128, 128))
                rgba = raw_img if raw_img.mode == "RGBA" else raw_img.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[3])
                image_np = np.array(bg)
            else:
                image_np = np.array(raw_img.convert("RGB"))
    elif isinstance(image_input, Image.Image):
        raw_img = ImageOps.exif_transpose(image_input)
        if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
            bg = Image.new("RGB", raw_img.size, (128, 128, 128))
            rgba = raw_img if raw_img.mode == "RGBA" else raw_img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            image_np = np.array(bg)
        else:
            image_np = np.array(raw_img.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        # Scale float values to uint8 [0, 255]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            image_np = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            image_np = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # Alpha composite onto neutral gray (128, 128, 128)
            rgb = arr[:, :, :3].astype(np.float32)
            alpha = (arr[:, :, 3:4].astype(np.float32)) / 255.0
            image_np = (rgb * alpha + 128.0 * (1.0 - alpha)).astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            image_np = arr
        else:
            raise ValueError(f"Unexpected numpy array shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
    transform = get_val_transforms(img_size=img_size, method=method)
    res = transform(image=image_np)["image"]

    if use_tta:
        # Multi-view TTA: original + horizontal flip
        flipped_np = np.ascontiguousarray(np.fliplr(image_np))
        res_flip = transform(image=flipped_np)["image"]
        if isinstance(res, torch.Tensor):
            stacked = torch.stack([res, res_flip], dim=0)
            return stacked if return_tensor else np.ascontiguousarray(stacked.cpu().numpy())
        else:
            stacked = np.stack([res, res_flip], axis=0)
            return torch.tensor(stacked) if return_tensor else np.ascontiguousarray(stacked)

    if return_tensor:
        if isinstance(res, torch.Tensor):
            return res.unsqueeze(0)
        else:
            return torch.tensor(res).unsqueeze(0)
    else:
        if isinstance(res, torch.Tensor):
            return np.ascontiguousarray(res.unsqueeze(0).cpu().numpy())
        else:
            return np.ascontiguousarray(np.expand_dims(res, 0))
