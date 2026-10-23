# Phase 4 Report: Preprocessing, Aspect Ratio & Augmentation Strategy

## 1. Preprocessing & Aspect Ratio Preservation Comparison

When resizing raw agricultural leaf photographs to the $224 \times 224$ network input, preserving natural aspect ratio is vital:
- **Direct Squash Resize**: Distorts blade morphology. Narrow elongate rice leaves become squished, changing thin longitudinal bacterial leaf blight streaks into oblong lesions.
- **Center-Crop Resize**:
  - Resizes the shorter edge to 256px, then crops the central $224 \times 224$ square.
  - **Pros**: Matches standard ImageNet transfer-learning paradigms, preserves natural lesion scale and fine cellular textures without zero-padding artifacts.
  - **Trade-off**: Peripheral lesions near leaf tips may be clipped.
- **Padded Resize (Letterbox / PadIfNeeded)**:
  - Scales the longest edge to 224px and pads border regions with black/reflect pixels.
  - **Pros**: Captures entire leaf perimeter.
  - **Trade-off**: Introduces artificial high-contrast borders and empty background pixels that CNN kernels may correlate with false signatures.
- **Selection**: We implement **Center-Crop** as standard for evaluation and deployment, while Albumentations' **RandomResizedCrop** during training forces the network to become scale- and aspect-invariant across varying crops.

---

## 2. Training Augmentation Suite (`albumentations`)

To prevent the model from overfitting to PlantVillage's sterile, uniform backgrounds and to promote domain generalization for outdoor field conditions, our augmentation pipeline incorporates:

| Transformation | Albumentations Op | Parameter Configuration | Agronomic / Robustness Rationale |
|:---|:---|:---|:---|
| **Crop & Scale** | `RandomResizedCrop` | scale=(0.75, 1.0) | Simulates varying camera distances and zoom levels |
| **Spatial Invariance** | `HorizontalFlip`, `VerticalFlip`, `RandomRotate90` | p=0.5 | Foliage can be photographed from any orientation |
| **Perspective** | `Affine` | scale=(0.95, 1.05), rotate=(-25, 25) | Simulates slight hand-held smartphone tilts in field |
| **Photometry** | `RandomBrightnessContrast`, `RandomGamma` | gamma=(80, 120), p=0.4-0.5 | Simulates overcast, midday harsh sun, and dusk lighting |
| **Color Shifts** | `HueSaturationValue` | hue $\pm 15$, sat $\pm 25$, val $\pm 20$ | Accounts for camera sensor color temperature variations |
| **Lens Blur** | `GaussianBlur`, `MotionBlur` | blur_limit=(3, 5), p=0.3 | Robustness against unfocused mobile photos and hand tremors |
| **Sensor Noise** | `GaussNoise` | std_range=(0.04, 0.12), p=0.3 | Handles low-light smartphone camera ISO noise |
| **Compression** | `ImageCompression` | quality=(65, 95), p=0.3 | Simulates WhatsApp/messaging compression artifacts |
| **Shadows** | `RandomShadow` | p=0.3 | Simulates shadows cast by farmer's hand or canopy |
| **Background Perturbation** | `CoarseDropout` | 1-4 holes, size 16-36px, p=0.4 | Prevents network from relying on solid background colors |

---

## 3. Zero Train/Serve Skew

All preprocessing routines reside in a single canonical shared module:
[`src/data/transforms.py`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task-222/src/data/transforms.py)

- `preprocess_image_for_inference()` is invoked identically inside the training validation loop, offline evaluation benchmarks, and the live FastAPI endpoint.
- Ensures identical pixel normalization ($mean=[0.485, 0.456, 0.406]$, $std=[0.229, 0.224, 0.225]$), channel ordering (RGB), and tensor shape across training and production serving.
