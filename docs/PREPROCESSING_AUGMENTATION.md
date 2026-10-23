# Preprocessing, Normalization & Data Augmentation Pipeline

> **Disclaimer**: *AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.*

---

## 1. Zero Train/Serve Skew Design

A central failure mode in production computer vision is discrepancy between training preprocessing and production inference preprocessing. In this system:
- A single canonical Python module ([`src/data/transforms.py`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task-222/src/data/transforms.py)) governs image decoding, resizing, and normalization across:
  1. Offline Model Training (PyTorch)
  2. Test Set Evaluation & Benchmarking
  3. Live REST API Serving (FastAPI + ONNX Runtime)

---

## 2. Shared Canonical Preprocessing Specifications

- **Input Format**: 3-channel RGB image (automatic channel expansion for grayscale, alpha-strip for RGBA).
- **Spatial Resolution**: $224 \times 224$ pixels (standard EfficientNet/MobileNet receptive field).
- **Aspect Ratio Preservation**:
  - Image is first resized to $256 \times 256$ pixels.
  - A deterministic $224 \times 224$ center crop is extracted, eliminating boundary stretch distortion.
- **Color Normalization**:
  - Pixel intensities mapped to $[0.0, 1.0]$.
  - Standard ImageNet channel-wise normalization:
    $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 3. Training Augmentation Pipeline (Albumentations)

To prevent overfitting on laboratory solid backdrops and improve resilience against mobile camera artifacts, training uses a rich, stochastic pipeline:

| Augmentation Operator | Parameters / Range | Agricultural Motivation |
| :--- | :--- | :--- |
| `RandomResizedCrop` | Scale: 0.75–1.0, Size: 224x224 | Simulates varying camera-to-leaf distances |
| `HorizontalFlip` & `VerticalFlip` | Probability: 0.5 each | Foliar lesions are orientation-invariant |
| `RandomRotate90` & `Affine` | Rotation: $\pm 25^\circ$, Scale: 0.95–1.05 | Handheld smartphone tilt and rotation |
| `RandomBrightnessContrast` | Limit: $\pm 20\%$, Probability: 0.5 | Variable field sunlight and overcast clouds |
| `HueSaturationValue` | Hue: $\pm 15$, Sat: $\pm 25$, Val: $\pm 20$ | Cultivar color variations and sensor ISP styles |
| `GaussianBlur` / `MotionBlur` | Blur limit: 3–5 px, Probability: 0.3 | Camera shake and slight autofocus drift |
| `GaussNoise` | Std range: 0.04–0.12, Probability: 0.3 | Sensor ISO thermal noise in dim lighting |
| `ImageCompression` | JPEG quality: 65–95, Probability: 0.3 | WhatsApp/Telegram compression artifacts |
| `RandomShadow` | Shadow count: 1–3, Probability: 0.3 | Grower's hand or branch shadows over leaf |
| `CoarseDropout` | 1–4 holes, 16–36px size, Probability: 0.4 | Prevents background and table overfitting |

---

## 4. Validation & Production Inference Pipeline

No random or stochastic transformations are applied during validation or inference:
```python
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```
This guarantees that repeated queries with the exact same image produce 100% deterministic prediction outputs and calibrated confidences.
