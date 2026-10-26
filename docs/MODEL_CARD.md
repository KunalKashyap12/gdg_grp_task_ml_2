# Model Card: MobileNetV3-Large Crop Leaf Disease Classifier

> [!IMPORTANT]
> **Mandatory Agricultural & Safety Disclaimer**:  
> **AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.**  
> Critical decisions regarding chemical spraying, crop culling, or quarantine must be validated by trained agronomists, plant pathologists, or certified agricultural extension officers.

---

## 1. Model Details

- **Model Identifier**: MobileNetV3-Large Foliar Pathology Classifier
- **Production Version**: `1.1.0` (with `1.0.0` preserved as fallback in `models_backup_v1.0.0/`)
- **Architecture**: MobileNetV3-Large with hardswish activations, squeeze-and-excitation attention blocks, and modified linear projection head.
- **Parameters**: 4.25 Million parameters (~16.2 MB ONNX / 17.2 MB PyTorch checkpoint).
- **Input Specifications**: RGB Tensor of shape `[B, 3, 224, 224]`, normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
- **Output Architecture**:
  - *v1.0.0*: 37 class logits.
  - *v1.1.0*: 38 class logits (37 supported botanical classes + 1 internal reject class `Unsupported___Unsupported_crop` for out-of-scope foliage rejection).
- **Confidence Calibration**: Post-hoc Temperature Scaling ($T$) fitted on held-out validation sets via negative log-likelihood minimization.
- **Safety Thresholding**: Configured confidence cutoff $\theta$, margin threshold ($0.15$), and predictive entropy filter ($1.50$) to reject ambiguous or unsupported leaves.
- **Runtime Engines**: PyTorch 2.14+, ONNX Runtime 1.23+ (Opset 14/17).
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

## 4. Quantitative Performance Metrics & Version Comparison

### Benchmark Comparison: v1.0.0 vs v1.1.0

| Evaluation Metric | Baseline v1.0.0 | Retrained v1.1.0 | Absolute Delta | Relative Change | Gate Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lab Test Accuracy** | 89.86% | **96.43%** | +6.57% pts | +7.3% | <= -2.0% drop | **PASS** |
| **Lab Test Macro F1** | 87.77% | **94.97%** | +7.20% pts | +8.2% | Preserve quality | **PASS** |
| **Field Test Accuracy (PlantDoc)** | 15.48% | **47.68%** | +32.20% pts | +208.0% | > 0.0% gain | **PASS** |
| **Field Selective Accuracy** | 23.85% | **68.66%** | +44.80% pts | +187.8% | High precision | **PASS** |
| **Field Test Coverage** | 33.75% | **41.49%** | +7.74% pts | +22.9% | Diagnostic yield | **PASS** |
| **Overall False-Healthy Rate** | 1.96% | **0.47%** | -1.49% pts | -76.2% | <= 0.0% increase | **PASS** |
| **Potato False-Healthy Rate** | 4.44% | **0.56%** | -3.89% pts | -87.5% | <= 0.0% increase | **PASS** |
| **Rice False-Healthy Rate** | 16.25% | **2.50%** | -13.75% pts | -84.6% | <= 0.0% increase | **PASS** |
| **OOD Conf > 0.80 Fraction** | 27.69% | **0.00%** | -27.69% pts | -100.0% | Substantial drop | **PASS** |
| **OOD Accepted at Threshold** | 35.38% | **0.00%** | -35.38% pts | -100.0% | Minimize hallucinations | **PASS** |
| **OOD Identified as Unsupported** | 0.00% | **86.15%** | +86.15% pts | +0.0% | Internal reject class | **PASS** |

---

## 5. Confidence Calibration & Thresholding (Platt Scaling)

- **Post-Hoc Temperature Scaling (v1.1.0)**:
  - Optimized temperature: **$T = 0.6517$** fitted on the mixed validation set (laboratory + outdoor field + reject classes).
  - Pre-calibration Validation NLL: 0.4796 $\to$ Post-calibration Validation NLL: **0.3878** (0.0917 reduction).
- **Operating Low-Confidence Threshold ($\theta = 0.70$)**:
  - Calibrated on the mixed validation set: achieves **96.36% selective accuracy** at **84.22% coverage** (and 96.97% selective accuracy at 82.70% coverage for $\theta = 0.74$).
  - Any prediction with confidence $< \theta$, top margin $< 0.15$, or entropy $> 1.50$ safely returns status `"low_confidence"` or `"unsupported_crop"`.
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
