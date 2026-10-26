# Model Limitations, Operational Boundaries & Risk Analysis

> [!IMPORTANT]
> **Mandatory Disclaimer**:  
> **AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.**  
> Never initiate costly pesticide interventions, crop destruction, or quarantine measures based solely on computer vision predictions.

---

## 1. Laboratory-to-Field Domain Shift (The Reality of Field Photography)

- **Observed Empirical Drop in v1.0.0**: While the baseline model achieved **88.41% accuracy on controlled test data**, accuracy dropped sharply when evaluated directly against **PlantDoc** uncurated outdoor field imagery due to complex field backgrounds, harsh sunlight, and leaf occlusions.
- **v1.1.0 Field Data Enrichment**: In model version `1.1.0`, 3,148 field crop samples (partitioned with zero data leakage from official PlantDoc training scenes) and heavy domain-robust augmentations (ColorJitter, Blur, Affine, CoarseDropout) were integrated directly into the training curriculum, significantly lifting field diagnostic resilience.
- **Operational Recommendation**: Users should still be instructed to isolate a single leaf against a neutral surface or their palm whenever possible to approximate optimal diagnostic illumination.

---

## 2. Closed-Set Scope Limitation & Out-of-Scope Mitigation (10 Supported Crops)

- The model officially supports **10 agricultural crops across 37 pathological classes**:
  - Tomato, Potato, Pepper (bell), Maize, Apple, Grape, Peach, Strawberry, Cherry, and Rice.
- **Vulnerability in v1.0.0**: When an unsupported crop (e.g., soybean, squash, blueberry, raspberry, wheat, or non-crop weeds) was presented, closed-set softmax forced assignment to one of the 37 supported classes.
- **v1.1.0 Dedicated Reject Architecture**:
  - Model `1.1.0` introduces a dedicated internal 38th output class (`Unsupported___Unsupported_crop`) trained on representative non-target crop foliage.
  - The API actively inspects the 38th class probability, top-1/top-2 margin ($< 0.15$), and predictive entropy ($> 1.50$), safely returning status `"unsupported_crop"` or `"low_confidence"` to prevent silent false-positive hallucinations on unlisted plant varieties.

---

## 3. Single-Leaf Constraint (No Canopy, Whole-Plant, or Root Triage)

- The model is trained exclusively on **individual leaf close-up photography**.
- It is incapable of diagnosing whole-canopy wilting, root rot, stem bores, or tuber damage.
- Submitting panoramic field images or whole-plant shots will degrade accuracy significantly.

---

## 4. Symptom Mimicry & Morphological Overlap Across Pathogens

- Different pathogens often manifest identical visual foliar damage:
  - *Tomato Early Blight* (fungal) vs. *Tomato Septoria Leaf Spot* (fungal) vs. *Tomato Bacterial Spot* (bacterial) produce visually overlapping necrotic spots during early infection stages.
  - Two-spotted spider mite damage mimics foliar viral chlorosis.
- **Diagnostic Advice**: The model provides top-3 candidate diagnoses (`top_k`). Always inspect secondary candidates and look for signs like fungal spores or bacterial ooze in a laboratory assay.

---

## 5. Multiple Simultaneous Infections

- Agricultural crops in open fields frequently suffer from co-occurring stresses (e.g., viral leaf curl combined with bacterial spot and nitrogen deficiency).
- The current architecture outputs a single multi-class distribution rather than multi-label pathology. The predicted label corresponds to the most dominant visual lesion, potentially masking underlying co-infections.

---

## 6. Abiotic Stress & Nutrient Deficiency Confounds

- Non-pathological abiotic stresses commonly mimic biotic diseases:
  - Potassium or magnesium deficiency creates marginal leaf chlorosis that closely resembles late blight or viral yellowing.
  - Chemical/pesticide burns manifest as necrotic foliar lesions indistinguishable from fungal spots.
- Visual AI alone cannot differentiate chemical scorch from biological infection without soil chemistry and spraying history.

---

## 7. Geographic, Cultivar & Variety Bias

- Datasets were primarily collected across North American and European agricultural cultivars (PlantVillage) and specific Philippine paddy strains.
- Indigenous landraces, heirloom crop varieties, or tropical cultivars grown in South Asia or Sub-Saharan Africa may display distinct leaf morphology or atypical lesion pigmentation.

---

## 8. Camera, Sensor & Lighting Sensitivity

- As demonstrated in Phase 8 synthetic stress benchmarks:
  - **Overexposure / Direct Sun Glare**: Causes an accuracy drop of over **-40%**, washing out light-colored fungal mycelia.
  - **Severe Motion / Defocus Blur**: Degrades accuracy by **-10% to -35%**, smoothing out spot margins.
  - **Low Sensor Resolution (<72px)**: Yields a **-54% drop**, destroying micro-texture.
- **Guidance**: Always capture photos in diffused daylight, tap to focus on the lesion, and ensure the leaf fills at least 70% of the frame.
