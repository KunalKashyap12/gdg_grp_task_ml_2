# Model Improvement & Acceptance Gate Evaluation Report: v1.0.0 vs v1.1.0

**Evaluation Date**: 2026-10-26 20:57:39
**Gate Decision**: **ACCEPTED**

## 1. Executive Summary

- Baseline Model: MobileNetV3-Large v1.0.0 (37 classes, $T=0.4792$, $\theta=0.74$)
- Retrained Model: MobileNetV3-Large v1.1.0 (38 classes [37 supported + 1 internal reject], $T=0.6517$, $\theta=0.7$)
- Final Gate Verdict: **ACCEPTED**

## 2. Quantitative Comparison Table

| Evaluation Metric | Baseline v1.0.0 | Retrained v1.1.0 | Absolute Delta | Relative Change | Gate Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lab Test Accuracy** | 89.86% | 96.43% | +6.57% pts | +7.3% | <= -2.0% drop | **PASS** |
| **Lab Test Macro F1** | 87.77% | 94.97% | +7.20% pts | +8.2% | Preserve quality | **PASS** |
| **Field Test Accuracy (PlantDoc)** | 15.48% | 47.68% | +32.20% pts | +208.0% | > 0.0% gain | **PASS** |
| **Field Selective Accuracy** | 23.85% | 68.66% | +44.80% pts | +187.8% | High precision | **PASS** |
| **Field Test Coverage** | 33.75% | 41.49% | +7.74% pts | +22.9% | Diagnostic yield | **PASS** |
| **Overall False-Healthy Rate** | 1.96% | 0.47% | -1.49% pts | -76.2% | <= 0.0% increase | **PASS** |
| **Potato False-Healthy Rate** | 4.44% | 0.56% | -3.89% pts | -87.5% | <= 0.0% increase | **PASS** |
| **Rice False-Healthy Rate** | 16.25% | 2.50% | -13.75% pts | -84.6% | <= 0.0% increase | **PASS** |
| **OOD Conf > 0.80 Fraction** | 27.69% | 0.00% | -27.69% pts | -100.0% | Substantial drop | **PASS** |
| **OOD Accepted at Threshold** | 35.38% | 0.00% | -35.38% pts | -100.0% | Minimize hallucinations | **PASS** |
| **OOD Identified as Unsupported** | 0.00% | 86.15% | +86.15% pts | +0.0% | Internal reject class | **PASS** |

## 3. Per-Class Recall Comparison (All 37 Classes)

| Class ID | Class Label | v1.0.0 Recall | v1.1.0 Recall | Delta | Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 00 | `Tomato___Bacterial_spot` | 86.7% | 96.7% | +10.0% pts | IMPROVED |
| 01 | `Tomato___Early_blight` | 47.8% | 88.9% | +41.1% pts | IMPROVED |
| 02 | `Tomato___Late_blight` | 65.6% | 82.2% | +16.7% pts | IMPROVED |
| 03 | `Tomato___Leaf_mold` | 82.2% | 100.0% | +17.8% pts | IMPROVED |
| 04 | `Tomato___Septoria_leaf_spot` | 93.3% | 97.8% | +4.4% pts | STABLE |
| 05 | `Tomato___Spider_mites` | 90.0% | 97.8% | +7.8% pts | IMPROVED |
| 06 | `Tomato___Target_spot` | 74.4% | 97.8% | +23.3% pts | IMPROVED |
| 07 | `Tomato___Yellow_leaf_curl_virus` | 87.8% | 98.9% | +11.1% pts | IMPROVED |
| 08 | `Tomato___Mosaic_virus` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 09 | `Tomato___Healthy` | 98.9% | 100.0% | +1.1% pts | STABLE |
| 10 | `Potato___Early_blight` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 11 | `Potato___Late_blight` | 71.1% | 96.7% | +25.6% pts | IMPROVED |
| 12 | `Potato___Healthy` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 13 | `Pepper___Bacterial_spot` | 97.8% | 94.4% | -3.3% pts | STABLE |
| 14 | `Pepper___Healthy` | 95.6% | 98.9% | +3.3% pts | STABLE |
| 15 | `Maize___Gray_leaf_spot` | 96.2% | 96.2% | +0.0% pts | STABLE |
| 16 | `Maize___Common_rust` | 93.3% | 95.6% | +2.2% pts | STABLE |
| 17 | `Maize___Northern_leaf_blight` | 85.6% | 83.3% | -2.2% pts | STABLE |
| 18 | `Maize___Healthy` | 98.9% | 100.0% | +1.1% pts | STABLE |
| 19 | `Apple___Apple_scab` | 98.9% | 100.0% | +1.1% pts | STABLE |
| 20 | `Apple___Black_rot` | 93.3% | 100.0% | +6.7% pts | IMPROVED |
| 21 | `Apple___Cedar_apple_rust` | 100.0% | 97.6% | -2.4% pts | STABLE |
| 22 | `Apple___Healthy` | 93.3% | 97.8% | +4.4% pts | STABLE |
| 23 | `Grape___Black_rot` | 91.1% | 94.4% | +3.3% pts | STABLE |
| 24 | `Grape___Esca_black_measles` | 90.0% | 100.0% | +10.0% pts | IMPROVED |
| 25 | `Grape___Leaf_blight` | 97.8% | 100.0% | +2.2% pts | STABLE |
| 26 | `Grape___Healthy` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 27 | `Peach___Bacterial_spot` | 96.7% | 100.0% | +3.3% pts | STABLE |
| 28 | `Peach___Healthy` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 29 | `Strawberry___Leaf_scorch` | 95.6% | 97.8% | +2.2% pts | STABLE |
| 30 | `Strawberry___Healthy` | 98.6% | 100.0% | +1.4% pts | STABLE |
| 31 | `Cherry___Powdery_mildew` | 97.8% | 100.0% | +2.2% pts | STABLE |
| 32 | `Cherry___Healthy` | 100.0% | 100.0% | +0.0% pts | STABLE |
| 33 | `Rice___Bacterial_leaf_blight` | 65.5% | 86.2% | +20.7% pts | IMPROVED |
| 34 | `Rice___Brown_spot` | 51.7% | 58.6% | +6.9% pts | IMPROVED |
| 35 | `Rice___Leaf_blast` | 59.1% | 72.7% | +13.6% pts | IMPROVED |
| 36 | `Rice___Healthy` | 95.0% | 90.0% | -5.0% pts | STABLE |

## 4. Acceptance Gate Decision & Justification

### Final Verdict: **ACCEPTED**

### Detailed Criteria Breakdown:
1. **Lab Test Retention**: PASSED (Drop = -6.57%, limit <= 2.0%)
2. **Field Generalization**: PASSED (Accuracy changed from 15.48% to 47.68%, gain = +32.20%)
3. **Out-of-Scope Protection**: PASSED (High confidence hallucinations dropped by 27.69% pts)
4. **Diagnostic Safety (False-Healthy)**: PASSED (Potato FH diff: -3.89%, Rice FH diff: -13.75%)
5. **Recall Stability**: PASSED (0 classes dropped >10%)

## 5. Mandatory Disclaimer

> **Agricultural Safety Disclaimer**: *AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.*
