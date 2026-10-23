# Model Limitations, Operational Boundaries & Risk Analysis

> [!IMPORTANT]
> **Mandatory Disclaimer**:  
> **AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.**  
> Never initiate costly pesticide interventions, crop destruction, or quarantine measures based solely on computer vision predictions.

---

## 1. Laboratory-to-Field Domain Shift (The Reality of Field Photography)

- **Observed Empirical Drop**: While the model achieves **88.41% accuracy on controlled test data**, accuracy drops to **11.46%** when evaluated directly against **PlantDoc** uncurated outdoor field imagery.
- **Root Causes**:
  - Training data (PlantVillage) features detached single leaves photographed flat against uniform, solid-colored laboratory backdrops.
  - In real farming conditions, foliage is surrounded by soil, weeds, adjacent branches, and variable sky illumination.
- **Operational Recommendation**: Users must be instructed to isolate a single leaf against a neutral surface or their palm to approximate the training distribution.

---

## 2. Closed-Set Scope Limitation (Only 10 Crops & 37 Classes)

- The model operates as a closed-set softmax classifier over exactly **37 predefined classes across 10 crops**:
  - Tomato, Potato, Pepper (bell), Maize, Apple, Grape, Peach, Strawberry, Cherry, and Rice.
- **Vulnerability**: If an out-of-scope leaf (e.g., soybean, wheat, cotton, cassava, coffee, or wild weeds) is uploaded, the model will forcibly assign it to one of the 37 supported classes.
- **Mitigation**: The API enforces a calibrated threshold ($\theta = 0.74$) and entropy boundary to reject uncertain inputs with status `"low_confidence"`.

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
