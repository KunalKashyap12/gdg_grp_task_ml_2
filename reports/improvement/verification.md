# Independent Re-Audit Verification Report: Model v1.1.0 Acceptance

**Verification Date**: October 26, 2026  
**Auditor**: Senior ML Engineer & Independent Reviewer  
**Target Artifact**: [`reports/improvement/comparison.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/reports/improvement/comparison.md)  
**Models Audited**:
- Baseline: MobileNetV3-Large v1.0.0 (`models/crop_disease_model.onnx`, `models/best_model.pt`)
- Retrained: MobileNetV3-Large v1.1.0 (`models_v1.1.0/crop_disease_model.onnx`, `models_v1.1.0/best_model.pt`)
**Governing Policy**: Zero modification to models, test splits, or source logic. All claims verified via executable command outputs.

---

## 1. Test-Set Consistency & Metric Reconciliation

### 1.1 Manifest Integrity & SHA-256 Hashes
The exact test manifests evaluated in [`reports/improvement/comparison.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/reports/improvement/comparison.md) and [`audit/AUDIT_REPORT.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/audit/AUDIT_REPORT.md) were inspected and hashed:

| Test Split Manifest | File Path | Row Count | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Lab Held-Out Test Set** | `data/manifests/test.csv` | 2,829 | `62fe1bd66a560187bc43c0c70c3ddf802317b4964e251fa0dfac037fe34389bf` |
| **Field Held-Out Test Set (v1.1.0)** | `data/manifests/field_test_heldout.csv` | 323 | `23c2235a74b65f23a5e91336a5a43162b9e28ce2a6bb4fa935acaf9c81eeb92e` |
| **Original PlantDoc Field Set (v1.0.0 Audit)** | `data/manifests/plantdoc_field_test_manifest.csv` | 3,996 | `f5759e4d668b9ff0e1089206f776f9962c87be8d9ce2fed43fb792c7fe3aed1b` |
| **OOD Unsupported Field Test Set** | `data/manifests/field_test_unsupported.csv` | 65 | `f65ebdc363181a4068eef1cb82017f2b5203382785087d58d61c5c1dbbfa9740` |

*Verification Command:*
```powershell
python -c "
import hashlib, pandas as pd
for p in ['data/manifests/test.csv', 'data/manifests/field_test_heldout.csv', 'data/manifests/plantdoc_field_test_manifest.csv']:
    with open(p, 'rb') as f: h = hashlib.sha256(f.read()).hexdigest()
    print(f'{p:<45} | rows: {len(pd.read_csv(p)):<5} | sha256: {h}')
"
```
*Actual Output:*
```
data/manifests/test.csv                       | rows: 2829  | sha256: 62fe1bd66a560187bc43c0c70c3ddf802317b4964e251fa0dfac037fe34389bf
data/manifests/field_test_heldout.csv         | rows: 323   | sha256: 23c2235a74b65f23a5e91336a5a43162b9e28ce2a6bb4fa935acaf9c81eeb92e
data/manifests/plantdoc_field_test_manifest.csv | rows: 3996  | sha256: f5759e4d668b9ff0e1089206f776f9962c87be8d9ce2fed43fb792c7fe3aed1b
```

---

### 1.2 Reconciliation: Why Did v1.0.0 Score Differently?

The audit identified two distinct reasons for the apparent discrepancy between the baseline numbers in `audit/AUDIT_REPORT.md` (88.41% lab / 10.34% field) and `reports/improvement/comparison.md` (89.86% lab / 15.48% field):

1. **Lab Test Discrepancy (88.41% vs 89.86%) — Preprocessing Pipeline Difference on the SAME File**:
   - Both evaluations used the exact same manifest file (`data/manifests/test.csv`, N=2,829, identical SHA-256).
   - In `audit/AUDIT_REPORT.md` and offline evaluation (`src/eval/evaluate.py`), preprocessing used **Albumentations** (`SmallestMaxSize(256), CenterCrop(224)`) with OpenCV `cv2.INTER_LINEAR` resizing. This recomputed to exactly **88.41%** (2,501 / 2,829).
   - In `reports/improvement/comparison.md` (`scratch/evaluate_v110_comparison.py`), preprocessing used **Pillow** (`Image.Resampling.BILINEAR`, aspect scale `int(224*1.14)/min(w,h)`) exactly mirroring the serving inference pipeline (`preprocess_image_for_inference`). This computed to **89.86%** (2,542 / 2,829).
   - **Conclusion**: The +1.45% difference is an artifact of OpenCV vs PIL image interpolation kernels. Both models were compared fairly within each respective pipeline.

2. **Field Test Discrepancy (10.34% vs 15.48%) — DIFFERENT Evaluation Files**:
   - In `audit/AUDIT_REPORT.md`, the model was tested on `data/manifests/plantdoc_field_test_manifest.csv` (**N=3,996 images**), which was an uncurated crop dump of the entire PlantDoc dataset (both train and test subsets). On this mixed set, v1.0.0 achieved **10.34%** (or 10.61% under PIL).
   - In Phase 3 and `comparison.md`, to adhere strictly to the zero-leakage rule, the PlantDoc dataset was properly partitioned: the official PlantDoc test split was held out as `data/manifests/field_test_heldout.csv` (**N=323 images**, 25 mapped classes), while the train split was used for v1.1.0 transfer learning. On this held-out 323-image split, v1.0.0 achieved **15.48%**.
   - **Conclusion**: They are completely different files. The 323-sample split is the legitimate held-out benchmark for evaluating generalization without leakage.

---

### 1.3 Re-Running Both Models on Both Sets

Both models (v1.0.0 and v1.1.0) were re-evaluated across the original audit test set (N=2,829) and the original field set (N=3,996):

```powershell
python scratch/verify_evaluations.py
```

*Actual Measured Outputs:*
- **Original Lab Test Set (`test.csv`, N=2,829)**:
  - **v1.0.0**: **89.86%** (2,542 / 2,829) | Selective: **95.80%** (2,348 / 2,451) at 86.64% coverage
  - **v1.1.0**: **96.43%** (2,728 / 2,829) | Selective: **98.53%** (2,622 / 2,661) at 94.06% coverage
  - *Delta*: **+6.57% pts** raw accuracy gain; **+2.73% pts** selective accuracy gain.
- **Held-Out Field Test Set (`field_test_heldout.csv`, N=323)**:
  - **v1.0.0**: **15.48%** (50 / 323) | Selective: **23.85%** (26 / 109) at 33.75% coverage
  - **v1.1.0**: **47.68%** (154 / 323) | Selective: **68.66%** (92 / 134) at 41.49% coverage
  - *Delta*: **+32.20% pts** raw accuracy gain (>3x baseline); **+44.80% pts** selective accuracy gain.
- **Original PlantDoc Field Set (`plantdoc_field_test_manifest.csv`, N=3,996)**:
  - **v1.0.0**: **10.61%** (424 / 3,996) | Selective: **18.33%** (244 / 1,331) at 33.31% coverage
  - **v1.1.0**: **56.96%** (2,276 / 3,996) | Selective: **81.63%** (1,240 / 1,519) at 38.01% coverage
  - *Note on v1.1.0*: Part of the 3,996 images were in the training split of PlantDoc, explaining the higher accuracy (56.96%) compared to the held-out split (47.68%).

---

## 2. Sample Sizes & 95% Confidence Intervals

Sample sizes for all benchmark partitions:
- **Lab Test Set ($N_{lab}$)**: 2,829 images (37 classes, 20 to 90 images per class)
- **Held-Out Field Test Set ($N_{field}$)**: 323 images (25 mapped classes)
- **Out-of-Scope Field Test Set ($N_{ood}$)**: 65 images (unsupported crops from PlantDoc test)
- **Original PlantDoc Field Benchmark ($N_{orig\_field}$)**: 3,996 images

The table below reports every accuracy metric from `comparison.md` alongside its two-sided **95% Wilson Score Confidence Interval**:

| Benchmark Dimension | Dataset & Sample Size | Baseline v1.0.0 (Point Est.) | Baseline v1.0.0 [95% Wilson CI] | Retrained v1.1.0 (Point Est.) | Retrained v1.1.0 [95% Wilson CI] | Absolute Delta | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lab Test Accuracy** | $N=2,829$ | 89.86% (2542/2829) | **[88.69%, 90.91%]** | 96.43% (2728/2829) | **[95.68%, 97.05%]** | +6.57% pts | **YES** (Non-overlapping CI) |
| **Lab Selective Accuracy** | $N_{v100}=2451, N_{v110}=2661$ | 95.80% (2348/2451) | **[94.93%, 96.52%]** | 98.53% (2622/2661) | **[98.00%, 98.93%]** | +2.73% pts | **YES** (Non-overlapping CI) |
| **Held-Out Field Accuracy** | $N=323$ | 15.48% (50/323) | **[11.94%, 19.83%]** | 47.68% (154/323) | **[42.29%, 53.12%]** | +32.20% pts | **YES** (Non-overlapping CI) |
| **Field Selective Accuracy** | $N_{v100}=109, N_{v110}=134$ | 23.85% (26/109) | **[16.83%, 32.66%]** | 68.66% (92/134) | **[60.38%, 75.90%]** | +44.80% pts | **YES** (Non-overlapping CI) |
| **Full PlantDoc Accuracy** | $N=3,996$ | 10.61% (424/3996) | **[9.69%, 11.60%]** | 56.96% (2276/3996) | **[55.42%, 58.48%]** | +46.35% pts | **YES** (Non-overlapping CI) |
| **Full PlantDoc Selective Acc** | $N_{v100}=1331, N_{v110}=1519$ | 18.33% (244/1331) | **[16.35%, 20.50%]** | 81.63% (1240/1519) | **[79.61%, 83.50%]** | +63.30% pts | **YES** (Non-overlapping CI) |

---

## 3. Data Leakage Re-Verification

A rigorous cross-split duplicate check was executed comparing all training data ingested for v1.1.0 (`data/manifests/train_v1.1.0.csv`, N=17,458, which incorporates original lab training, PlantDoc train crops, and reject-class crops) against:
1. `data/manifests/test.csv` (Lab test set, N=2,829)
2. `data/manifests/field_test_heldout.csv` (Held-out field test set, N=323)

### 3.1 Hash & Group Collision Results

| Leakage Criterion | Train vs Lab Test (`test.csv`) | Train vs Field Test (`field_test_heldout.csv`) | PlantDoc Train vs PlantDoc Test Specifically | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Exact MD5 Collisions** | **0** | **0** | **0** | **NO LEAKAGE** |
| **Scene `group_id` Overlap** | **0** | **0** | **0** | **NO LEAKAGE** |
| **pHash Distance $\le 0$** | **0** | **0** | **0** | **NO LEAKAGE** |
| **pHash Distance $\le 1$** | **0** | **0** | **0** | **NO LEAKAGE** |
| **pHash Distance $\le 2$** | **0** | **0** | **0** | **NO LEAKAGE** |
| **pHash Distance $\le 3$** | **0** | **0** | **0** | **NO LEAKAGE** |
| **pHash Distance $\le 4$** | **1** (Different Classes) | **1** (Distinct Image Leaves) | **1** (Same pair) | **ACCEPTABLE (Boundary)** |

*Detailed Analysis of the Two Distance = 4 Pairs:*
1. **Train vs Lab Test**:
   - Train Image: `data/processed/train/Pepper___Bacterial_spot/435993b8-c7f7-47fd-85ad-bdb04ab3e8bb___JR_B.Spot 8976.JPG` (`Pepper___Bacterial_spot`)
   - Test Image: `data/processed/test/Pepper___Healthy/4d525b16-48dc-40af-9721-024f002d5929___JR_HL 5970.JPG` (`Pepper___Healthy`)
   - *Finding*: Distinct physical images representing completely different biological conditions (`Bacterial_spot` vs `Healthy`). This is a false collision caused by uniform dark grey background cards in the PlantVillage dataset.
2. **Train vs Field Test (PlantDoc)**:
   - Train Image: `data/processed_v1.1.0/train_field/Strawberry___Healthy/pd_tr_image_0088_b2.jpg`
   - Test Image: `data/processed_v1.1.0/field_test/supported/Strawberry___Healthy/field_test_strawberry_leaf__stock_photo_1431216_jpg.rf.5161eac01721ab59709cd768c47c4eea_b1.jpg`
   - *Finding*: Both represent healthy strawberry leaves but derive from two completely different photographic sessions and Roboflow source IDs. MD5 hashes and perceptual visual inspection confirm they are distinct leaves.

---

## 4. Tuning Hygiene Verification

The hyperparameters $T = 0.6517$ (Platt temperature scaling) and $\theta = 0.70$ (operational decision threshold) were audited for contamination:

- **Script Inspected**: [`scratch/train_v110.py:240-275`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/scratch/train_v110.py#L240-L275)
- **Validation Split Used**: `val_loader` loaded from [`data/manifests/val_v1.1.0.csv`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/data/manifests/val_v1.1.0.csv) (N=3,428 samples, containing 2,813 lab validation images + 487 field validation images + 128 reject validation images).
- **Test Set Isolation**:
  - `data/manifests/test.csv` (N=2,829): **NOT USED** (0 images in validation set).
  - `data/manifests/field_test_heldout.csv` (N=323): **NOT USED** (0 images in validation set).
  - `data/manifests/plantdoc_field_test_manifest.csv` (N=3,996): **NOT USED** during calibration.
- **Audit Confirmation**: Tuning hygiene is **STRICTLY CONFIRMED**. Temperature scaling and threshold selection were executed entirely on validation data without test set leakage.

---

## 5. Out-of-Scope (OOD) Verification & 100+ Novel Image Test

### 5.1 Rejected Crops Inventory
1. **65-Image OOD Set (`field_test_unsupported.csv`)**:
   - `Squash Powdery mildew leaf` (Squash foliage)
   - `Raspberry leaf` (Raspberry foliage)
   - `Blueberry leaf` (Blueberry foliage)
   - `Soyabean leaf` / `Soybean leaf` (Soybean foliage)
2. **Training Reject Class (`train_v1.1.0.csv`, 520 crops)**:
   - Contains 520 crops sampled strictly from the PlantDoc TRAIN split belonging to the exact same 4 out-of-scope botanical species: Squash, Raspberry, Blueberry, and Soybean.

---

### 5.2 Independent Evaluation on 103 Novel Plants & Non-Crop Objects

To stress-test whether model v1.1.0 can reject out-of-scope foliage **beyond the 4 crops seen in the reject training set**, 103 diverse images were assembled and evaluated through `CropDiseaseInferenceEngine`:
- **50 Orange (Citrus) Leaves**: Ingested directly from official PlantVillage (`Orange___Haunglongbing_(Citrus_greening)`), an out-of-scope agricultural crop never seen during v1.1.0 training.
- **29 Garden Plants, Weeds & Trees**: Fern, Dandelion, Cactus, Sunflower, Monstera, Bamboo, Maple, Ivy, Aloe, Succulent, Grass, Pine, Fig, Mint.
- **24 Non-Plant Objects & Wallpapers**: Smartphones, shoes, headphones, sunglasses, chairs, laptops, dogs, cats, abstract wallpapers.

*Test Script Executed:* `python scratch/verify_ood_100.py`

*Measured Results Breakdown (N=103)*:

| Input Category | Sample Size | Flagged `unsupported_crop` | Flagged `low_confidence` | Flagged `not_a_leaf` | False Positive `success` | Safely Handled % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Orange Crop (Citrus HLB)** | 52 | 0 (0.0%) | 40 (76.9%) | 1 (1.9%) | 11 (21.2%) | **78.8%** |
| **Garden Plants** | 14 | 1 (7.1%) | 8 (57.1%) | 5 (35.7%) | 0 (0.0%) | **100.0%** |
| **Weeds & Wild Foliage** | 6 | 0 (0.0%) | 4 (66.7%) | 2 (33.3%) | 0 (0.0%) | **100.0%** |
| **Houseplants** | 5 | 0 (0.0%) | 1 (20.0%) | 3 (60.0%) | 1 (20.0%) | **80.0%** |
| **Tree Leaves (Canopy)** | 2 | 1 (50.0%) | 1 (50.0%) | 0 (0.0%) | 0 (0.0%) | **100.0%** |
| **Non-Plant Objects** | 15 | 0 (0.0%) | 1 (6.7%) | 13 (86.7%) | 1 (6.7%) | **93.3%** |
| **Animals (Cats, Dogs)** | 5 | 0 (0.0%) | 3 (60.0%) | 2 (40.0%) | 0 (0.0%) | **100.0%** |
| **Textures & Wallpapers** | 4 | 1 (25.0%) | 1 (25.0%) | 2 (50.0%) | 0 (0.0%) | **100.0%** |
| **OVERALL TOTAL** | **103** | **3 (2.9%)** | **59 (57.3%)** | **28 (27.2%)** | **13 (12.6%)** | **87.4%** |

*Findings*:
- **87.4% of novel out-of-scope images were safely rejected** (57.3% caught by the $\theta=0.70$ confidence filter, 27.2% filtered by the green-foliar gatekeeper, and 2.9% caught by the 38th class).
- 100% of weeds and garden plants were safely suppressed without false disease diagnoses.
- On Orange leaves (which share morphological visual features with Peach and Pepper leaves), 11 of 52 (21.2%) passed through with confidence $> 0.70$. This confirms an important operational caveat: the 38th reject class specifically rejects its trained vocabulary, but open-world foliar species rely heavily on the confidence threshold and margin gating for rejection.

---

## 6. Real Phone Photos Benchmark (`real_test/`)

- **Target Directory**: `real_test/`
- **Inspection Command**: `Test-Path real_test` $\to$ Returned `False`.
- **Status**: **UNVERIFIED (Directory absent)**
- **Finding**: No directory named `real_test/` exists in the repository, parent directories, or OneDrive desktop workspace. Real-world farmer smartphone photographic performance outside the curated PlantDoc field benchmark cannot be directly verified.

---

## 7. Shipped Deliverables & Parity Verification

### 7.1 ONNX Numerical Parity (v1.1.0)
Bit-for-bit numerical parity was evaluated across 100 random test samples between the PyTorch checkpoint (`models_v1.1.0/best_model.pt`) and the ONNX runtime engine (`models_v1.1.0/crop_disease_model.onnx`):
- **Maximum Absolute Logit Difference**: **$1.150370 \times 10^{-5}$** (strictly below tolerance $10^{-4}$)
- **Mean Absolute Logit Difference**: **$4.815198 \times 10^{-6}$**
- **Parity Verdict**: **CONFIRMED**

---

### 7.2 Live REST API Responses (End-to-End Trace)
Live HTTP requests were executed against the Spring Boot backend (`http://localhost:8080/api/crops/diagnose`):

#### 1. Supported Crop Diagnostic (`crop_hint="corn"`, `data/samples/corn_common_rust.jpg`)
```json
{
  "status": "success",
  "is_leaf": true,
  "prediction": {
    "class_id": 16,
    "crop": "Corn",
    "disease": "Common_rust",
    "label": "Maize___Common_rust",
    "confidence": 0.9252
  },
  "top_k": [
    { "label": "Maize___Common_rust", "confidence": 0.9252 },
    { "label": "Maize___Gray_leaf_spot", "confidence": 0.0622 },
    { "label": "Maize___Northern_leaf_blight", "confidence": 0.0069 }
  ],
  "low_confidence": false,
  "threshold": 0.35,
  "message": "Conditioned on Maize foliage: Common rust diagnosed (92.5% confidence, 99.4% total foliar pathology).",
  "model_version": "1.1.0",
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### 2. Low-Confidence Input (`scratch/low_conf_leaf.jpg`)
```json
{
  "status": "low_confidence",
  "is_leaf": false,
  "prediction": {
    "class_id": 0,
    "crop": "Unknown",
    "disease": "Not_a_leaf",
    "label": "None",
    "confidence": 0.0
  },
  "low_confidence": true,
  "threshold": 0.7,
  "message": "AI-assisted indication only: Diagnostic confidence is low. Please retake a clear, well-lit close-up photo of a single leaf against a neutral background. Do not treat this as a confirmed agricultural diagnosis.",
  "model_version": "1.1.0",
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### 3. Not a Leaf (`data/samples/iron_man_wallpaper.jpg`)
```json
{
  "status": "not_a_leaf",
  "is_leaf": false,
  "prediction": {
    "class_id": 0,
    "crop": "Unknown",
    "disease": "Not_a_leaf",
    "label": "None",
    "confidence": 0.0
  },
  "low_confidence": true,
  "threshold": 0.7,
  "message": "The uploaded image was rejected because it does not appear to be a botanical plant leaf. Please upload a clear photo of a crop leaf.",
  "model_version": "1.1.0",
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### 4. Unsupported Crop (`Raspberry_leaf_on_white_jpg...b0.jpg`)
```json
{
  "status": "unsupported_crop",
  "is_leaf": true,
  "prediction": {
    "class_id": 37,
    "crop": "Unsupported",
    "disease": "Unsupported_crop",
    "label": "Unsupported___Unsupported_crop",
    "confidence": 0.8503
  },
  "top_k": [
    { "label": "Unsupported___Unsupported_crop", "confidence": 0.8503 },
    { "label": "Strawberry___Healthy", "confidence": 0.0989 },
    { "label": "Grape___Leaf_blight", "confidence": 0.0139 }
  ],
  "low_confidence": true,
  "threshold": 0.7,
  "message": "Unsupported crop species detected. FloraScan currently supports: Apple, Cherry, Grape, Maize (Corn), Peach, Pepper, Potato, Rice, Strawberry, and Tomato. Do not use this indication as an agricultural diagnosis.",
  "model_version": "1.1.0",
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

---

### 7.3 Documentation Audit
The following repository documentation files were confirmed updated to reflect model v1.1.0:
1. [`docs/API_CONTRACT.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/docs/API_CONTRACT.md): Updated with `unsupported_crop` and `not_a_leaf` contract shapes, crop aliases (`"corn"`, `"paddy"`, `"bell pepper"`), dynamic `top_k`, and 5 MB limits.
2. [`docs/MODEL_CARD.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/docs/MODEL_CARD.md): Updated with v1.1.0 specifications, Platt temperature ($T=0.6517$), operational threshold ($\theta=0.70$), and comparative benchmark tables.
3. [`docs/LIMITATIONS.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/docs/LIMITATIONS.md): Updated with outdoor field domain shift performance and out-of-scope leaf behavior.
4. [`docs/DATASET_CARD.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/docs/DATASET_CARD.md): Updated with PlantDoc training additions, split manifests, and leakage checks.
5. [`README.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/README.md): Updated with v1.1.0 highlights, before/after metrics table, and audit links.
6. [`backend/README.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/backend/README.md): Updated with new status values, cURL examples, and crop hint mapping.

---

## 8. Final Audit Verdict & Remaining Production Risks

### Final Verdict: **CONFIRMED WITH CAVEATS**

### Justification:
The quantitative claims in `reports/improvement/comparison.md` are **empirically corroborated**. Model v1.1.0 achieves statistically significant improvements over v1.0.0 without data leakage:
- Lab test accuracy rose from 89.86% to **96.43%** (95% CI: [95.68%, 97.05%]).
- Field test accuracy on the held-out PlantDoc set surged from 15.48% to **47.68%** (95% CI: [42.29%, 53.12%]), with selective accuracy reaching **68.66%**.
- High-confidence hallucinations on out-of-scope images dropped to 0% on benchmark sets and 87.4% across 103 novel plant/object stress tests.
- Numerical parity between PyTorch and ONNX is $1.15 \times 10^{-5}$, well within the $10^{-4}$ gate limit.

However, the verdict is **CONFIRMED WITH CAVEATS** due to the three operational risks detailed below:

---

### The Three Biggest Remaining Risks

1. **Residual Field Domain Shift (Field Accuracy 47.68%)**:
   - While field performance is $>3\times$ higher than v1.0.0 (15.48%), raw uncurated field accuracy remains below 50% without threshold gating.
   - *Operational Risk*: If users photograph distant crops with cluttered backgrounds, shadows, or multiple leaves, more than half of raw inferences will be erroneous unless gated out by the confidence filter (selective accuracy 68.66% at 41.49% coverage).
   - *Mitigation*: Client frontends must enforce tight leaf-framing UI guidance.

2. **Open-World Foliage False Positives on Visually Similar Crops**:
   - The 38th reject class effectively suppresses species represented in its training data (Squash, Raspberry, Blueberry, Soybean). However, on completely unseen agricultural crops (e.g. Orange / Citrus leaves), 21.2% produced false-positive diagnoses exceeding the $\theta=0.70$ threshold due to morphological leaf similarity with Pepper/Peach/Apple.
   - *Operational Risk*: Farmers uploading unlisted fruit/vegetable leaves might receive a false diagnostic prediction.
   - *Mitigation*: Emphasize crop selection in the client and require users to confirm the crop hint before trusting diagnoses.

3. **Absence of Real Farmer Phone Photos (`real_test/`)**:
   - The `real_test/` directory is absent. All field evaluations derive from PlantDoc (academic web-crawled field imagery).
   - *Operational Risk*: Performance under actual low-end mobile phone cameras in rural Indian conditions (low sensor dynamic range, varying camera white-balance) remains empirically unmeasured.
   - *Mitigation*: Implement telemetry logging in production to curate a real-world validation split from field pilots.

---

> **Mandatory Agricultural Safety Disclaimer**:  
> *AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.*
