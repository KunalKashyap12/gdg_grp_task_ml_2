# Model Card: MobileNetV3-Large Crop Leaf Disease Classifier

> [!IMPORTANT]
> **Mandatory Agricultural & Safety Disclaimer**:  
> **AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.**  
> Critical decisions regarding chemical spraying, crop culling, or quarantine must be validated by trained agronomists, plant pathologists, or certified agricultural extension officers.

---

## 1. Model Details

- **Model Identifier**: MobileNetV3-Large Foliar Pathology Classifier
- **Version**: `1.0.0`
- **Architecture**: MobileNetV3-Large with hardswish activations, squeeze-and-excitation attention blocks, and modified linear projection head.
- **Parameters**: 4.25 Million parameters (16.20 MB ONNX / 16.40 MB PyTorch).
- **Input Specifications**: RGB Tensor of shape `[B, 3, 224, 224]`, normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
- **Output**: 37 class logits scaled via Platt Temperature Scaling ($T = 0.4792$) and filtered via low-confidence cutoff ($\theta = 0.74$).
- **Runtime Engines**: PyTorch 2.14+, ONNX Runtime 1.31+ (Opset 17).
- **License**: Apache 2.0.

---

## 2. Intended Use & Scope

### Primary Intended Uses:
- First-line agricultural decision support and mobile triage for smallholder farmers and extension officers across 10 vital crops: **Tomato, Potato, Pepper (bell), Maize, Apple, Grape, Peach, Strawberry, Cherry, and Rice**.
- Early detection of foliar fungal, bacterial, viral, and mite damage from smartphone leaf close-ups.
- Low-latency edge inference (<5ms on CPU) and containerized REST microservices.

### Out-of-Scope & Prohibited Use:
- **Autonomous Chemical Spraying**: The model must never trigger automated spraying without human agronomist oversight.
- **Unsupported Crops**: Images of wheat, cotton, cassava, soybean, or non-crop weeds will yield uninformative predictions and should be rejected via the confidence threshold.
- **Whole-Plant or Aerial Imagery**: The model is trained exclusively on individual leaf close-ups. Panoramic drone or root imagery is unsupported.
- **Definitive Clinical Quarantine**: Molecular laboratory assays (PCR/ELISA) are mandatory for definitive pathogen strain typing.

---

## 3. Training & Optimization Setup

- **Two-Stage Transfer Learning**:
  - *Stage 1*: Classifier head trained for 2 epochs with backbone frozen ($\text{lr} = 10^{-3}$, AdamW).
  - *Stage 2*: Full network fine-tuned with Cosine Annealing learning rate schedule ($\text{lr} = 10^{-4}$, AdamW, weight decay $10^{-4}$).
- **Objective Function**: Inverse class-frequency weighted cross-entropy loss with label smoothing ($\alpha = 0.1$).
- **Data Regularization**: Albumentations pipeline including RandomResizedCrop, flips, rotations, photometry, blur, sensor noise, JPEG compression, random shadows, and coarse cutout dropout.

---

## 4. Quantitative Performance Metrics

### Held-Out Leakage-Free Test Set (2,829 images, 37 classes)

| Evaluation Metric | Score | Note |
| :--- | :--- | :--- |
| **Top-1 Overall Accuracy** | **88.41%** | Exact class identification across 37 classes |
| **Top-3 Accuracy** | **97.74%** | Correct diagnosis contained in top-3 candidates |
| **Crop Identification Accuracy** | **96.29%** | Botanical species discrimination (e.g. Potato vs Pepper) |
| **Disease Discrimination Accuracy** | **89.36%** | Pathological condition discrimination |
| **Macro F1-Score** | **86.19%** | Unweighted mean across all 37 classes |
| **Weighted F1-Score** | **88.11%** | Class-weighted average |
| **CPU Latency (ONNX Runtime)** | **4.68 ms** | Sub-5ms latency on CPU (4.96x speedup over PyTorch) |

---

## 5. Confidence Calibration & Thresholding (Platt Scaling)

- **Post-Hoc Temperature Scaling**: Optimized strictly positive temperature $T = 0.4792$ on validation logits via L-BFGS with strong-Wolfe line search.
- **Expected Calibration Error (ECE)**:
  - *Validation Set*: Reduced from 21.54% down to **1.80%** (**91.6% relative reduction**).
  - *Held-Out Test Set*: Reduced from 20.81% down to **1.95%** (**90.6% relative reduction**).
- **Operating Low-Confidence Threshold ($\theta = 0.74$)**:
  - Calibrated on validation set to achieve **$\ge 95\%$ accuracy on accepted predictions**.
  - Held-out test set performance: **95.09% selective accuracy** at **84.23% coverage**.
  - Any prediction with confidence $< 0.74$ returns status `"low_confidence"` with advice to retake a clear photo.

---

## 6. Robustness, Synthetic Stress Testing & Field Domain Shift

### PlantDoc Real-Field Benchmark (3,996 images, 25 classes):
- **Raw Top-1 Accuracy**: **11.46%** (a drop of **-76.95% points** versus clean lab imagery).
- **With Low-Confidence Filter**: The $\theta = 0.74$ threshold correctly flags **66.12% of uncertain field samples as low-confidence**, elevating selective accuracy on accepted field images to **18.39%**.

### Synthetic Stress Tests (12 Perturbations):
- **High Resilience**: Severe rotation (+1.18%), Shadows (-0.34%), Gaussian blur (-0.68%), Heavy JPEG (-2.87%), Low brightness (-2.87%), Cluttered background (-3.89%).
- **High Vulnerability**: Heavy sensor noise (-57.26%), Downsampling to 72px (-54.56%), Overexposure (-40.20%), Partial leaf crop 50% occlusion (-38.18%).

---

## 7. Interpretability (Grad-CAM Saliency)

- Grad-CAM saliency maps from the final convolutional block (`model.features[-1]`) demonstrate that model activations concentrate directly on foliar necrotic lesions, chlorotic rings, and fungal pustules.
- Saliency outside the leaf lamina (solid background cards, table textures, fingers) is near zero, verifying that the model has not overfitted to laboratory collection artifacts.
