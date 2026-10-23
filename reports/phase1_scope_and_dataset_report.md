# Phase 1 Report: Scope Definition & Dataset Quality Audit

**Project**: End-to-End Calibrated Crop Disease Classification & REST API  
**Phase**: 1 – Scope: 10 Crops and Their Pathology Classes  
**Author**: Senior Machine Learning Engineer  
**Date**: October 2026  

---

## 1. Executive Summary

In Phase 1, we established the foundational taxonomy, dataset acquisition pipelines, deduplication heuristics, and class schema for 10 crops spanning 37 classes. 

- **10 Crops Supported**: Tomato, Potato, Bell Pepper, Maize, Apple, Grape, Peach, Strawberry, Cherry, and Rice.
- **37 Granular Classes**: Every crop incorporates a confirmed `Healthy` class and 1 to 9 distinct disease classes.
- **18,803 Total Cleaned Images**: Capped at 600 images per class for balanced training while retaining demographic richness.
- **Quality Assurance**: Every class contains $\ge 100$ verified images post-deduplication; zero classes dropped or marked "low-reliability".
- **Rice Sourcing & Style Note**: Rice leaves originate from a separate open dataset (GNU AGPL 3.0) captured in natural field settings, contrasting with the lab conditions of PlantVillage.

---

## 2. Agricultural Justification & Relevance to Indian Farmers

India possesses 15 distinct agro-climatic zones with diverse agrarian micro-economies. The 10 supported crops represent staple food security, major vegetable cash crops, and vital horticultural exports:

1. **Rice (*Oryza sativa*)**:
   - **Economic Relevance**: India is the world's second-largest rice producer and largest exporter. Rice occupies over 45 million hectares across West Bengal, Uttar Pradesh, Punjab, Andhra Pradesh, and Tamil Nadu.
   - **Disease Impact**: Bacterial leaf blight (*Xanthomonas oryzae*) and Rice blast (*Magnaporthe oryzae*) cause annual yield losses between 10% and 50% if unmanaged.

2. **Tomato (*Solanum lycopersicum*)**:
   - **Economic Relevance**: Critical daily vegetable grown across Karnataka, Andhra Pradesh, Madhya Pradesh, and Maharashtra.
   - **Disease Impact**: Early blight (*Alternaria*), late blight, and Yellow Leaf Curl Virus (transmitted by *Bemisia tabaci*) routinely wipe out entire standing crops within days of onset.

3. **Potato (*Solanum tuberosum*)**:
   - **Economic Relevance**: Core rabi crop across the Indo-Gangetic plains (UP, Bihar, WB).
   - **Disease Impact**: Late blight (*Phytophthora infestans*) is historically the most devastating disease in cooler hilly and northern plains regions.

4. **Maize / Corn (*Zea mays*)**:
   - **Economic Relevance**: Third most critical cereal crop in India, used heavily for poultry feed, starch production, and direct consumption across Karnataka, Rajasthan, MP, and Bihar.
   - **Disease Impact**: Northern leaf blight and common rust cause premature leaf desiccation and yield reduction.

5. **Bell Pepper / Capsicum (*Capsicum annuum*)**:
   - **Economic Relevance**: High-value greenhouse and protected-cultivation cash crop popular among progressive farmers in Maharashtra, Karnataka, and Himachal Pradesh.
   - **Disease Impact**: Bacterial spot spreads aggressively in warm, humid polyhouses.

6. **Apple (*Malus domestica*)**:
   - **Economic Relevance**: Economic backbone of Jammu & Kashmir and Himachal Pradesh farmers.
   - **Disease Impact**: Apple scab (*Venturia inaequalis*) leads to premature defoliation and severe fruit deformation.

7. **Grape (*Vitis vinifera*)**:
   - **Economic Relevance**: Lucrative export crop centered in Nashik and Sangli (Maharashtra) and parts of Karnataka and Andhra Pradesh.
   - **Disease Impact**: Black rot and Esca complex inflict severe dieback and vine decline.

8. **Peach, Strawberry & Cherry (*Prunus & Fragaria spp.*)**:
   - **Economic Relevance**: Rapidly growing high-remuneration temperate and polyhouse crops across Northern Himalayan belts and Mahabaleshwar (Maharashtra).
   - **Disease Impact**: Powdery mildew and leaf scorch undermine photosynthetic capacity and market value.

---

## 3. Class Taxonomy and Data Availability

The mapping configuration (`config.yaml`) and exported registry (`data/supported_classes.json`) standardize class IDs from 0 to 36:

| Class ID | Standard Label (`Crop___Disease`) | Crop | Disease | Clean Sample Count | Reliability Status |
|:---:|:---|:---|:---|:---:|:---:|
| 0 | `Tomato___Bacterial_spot` | Tomato | Bacterial Spot | 600 | Reliable |
| 1 | `Tomato___Early_blight` | Tomato | Early Blight | 600 | Reliable |
| 2 | `Tomato___Late_blight` | Tomato | Late Blight | 600 | Reliable |
| 3 | `Tomato___Leaf_mold` | Tomato | Leaf Mold | 600 | Reliable |
| 4 | `Tomato___Septoria_leaf_spot` | Tomato | Septoria Leaf Spot | 600 | Reliable |
| 5 | `Tomato___Spider_mites` | Tomato | Two-Spotted Spider Mites | 600 | Reliable |
| 6 | `Tomato___Target_spot` | Tomato | Target Spot | 600 | Reliable |
| 7 | `Tomato___Yellow_leaf_curl_virus` | Tomato | Yellow Leaf Curl Virus | 600 | Reliable |
| 8 | `Tomato___Mosaic_virus` | Tomato | Mosaic Virus | 372 | Reliable |
| 9 | `Tomato___Healthy` | Tomato | Healthy | 600 | Reliable |
| 10 | `Potato___Early_blight` | Potato | Early Blight | 600 | Reliable |
| 11 | `Potato___Late_blight` | Potato | Late Blight | 600 | Reliable |
| 12 | `Potato___Healthy` | Potato | Healthy | 152 | Reliable |
| 13 | `Pepper___Bacterial_spot` | Pepper | Bacterial Spot | 600 | Reliable |
| 14 | `Pepper___Healthy` | Pepper | Healthy | 600 | Reliable |
| 15 | `Maize___Gray_leaf_spot` | Maize | Gray Leaf Spot | 513 | Reliable |
| 16 | `Maize___Common_rust` | Maize | Common Rust | 600 | Reliable |
| 17 | `Maize___Northern_leaf_blight` | Maize | Northern Leaf Blight | 600 | Reliable |
| 18 | `Maize___Healthy` | Maize | Healthy | 600 | Reliable |
| 19 | `Apple___Apple_scab` | Apple | Apple Scab | 600 | Reliable |
| 20 | `Apple___Black_rot` | Apple | Black Rot | 600 | Reliable |
| 21 | `Apple___Cedar_apple_rust` | Apple | Cedar Apple Rust | 275 | Reliable |
| 22 | `Apple___Healthy` | Apple | Healthy | 600 | Reliable |
| 23 | `Grape___Black_rot` | Grape | Black Rot | 600 | Reliable |
| 24 | `Grape___Esca_black_measles` | Grape | Esca (Black Measles) | 600 | Reliable |
| 25 | `Grape___Leaf_blight` | Grape | Leaf Blight | 600 | Reliable |
| 26 | `Grape___Healthy` | Grape | Healthy | 422 | Reliable |
| 27 | `Peach___Bacterial_spot` | Peach | Bacterial Spot | 600 | Reliable |
| 28 | `Peach___Healthy` | Peach | Healthy | 360 | Reliable |
| 29 | `Strawberry___Leaf_scorch` | Strawberry | Leaf Scorch | 600 | Reliable |
| 30 | `Strawberry___Healthy` | Strawberry | Healthy | 454 | Reliable |
| 31 | `Cherry___Powdery_mildew` | Cherry | Powdery Mildew | 600 | Reliable |
| 32 | `Cherry___Healthy` | Cherry | Healthy | 600 | Reliable |
| 33 | `Rice___Bacterial_leaf_blight` | Rice | Bacterial Leaf Blight | 184 | Reliable |
| 34 | `Rice___Brown_spot` | Rice | Brown Spot | 190 | Reliable |
| 35 | `Rice___Leaf_blast` | Rice | Leaf Blast | 138 | Reliable |
| 36 | `Rice___Healthy` | Rice | Healthy | 143 | Reliable |

---

## 4. Duplicate Filtering & Data Integrity

- **Perceptual Hash Deduplication**: Using `imagehash.phash()` with Hamming distance $\le 4$, we discovered and removed 340 near-identical duplicate frames. This was especially prevalent in the rice field capture series where continuous burst shots had been archived.
- **Corrupted Image Rejection**: 0 corrupted files encountered in the final ingestion step.
- **Partition Integrity**: Stratified 70/15/15 split yields:
  - **Train partition**: 13,157 images
  - **Validation partition**: 2,815 images
  - **Test partition**: 2,831 images
- **Leakage Check**: Set intersection between Train, Val, and Test file paths is identically zero.

---

## 5. Verification Check Summary

All Phase 1 unit and integration tests passed via `pytest tests/test_phase1.py`:
- `test_config_crops`: PASSED (10 crops declared and validated)
- `test_supported_classes_json`: PASSED (37 classes conforming to schema)
- `test_dataset_audit_and_counts`: PASSED (Every class $\ge 100$ clean images)
- `test_manifest_partition_integrity`: PASSED (Zero data leakage across splits)
- `test_processed_image_folders`: PASSED (37 class directories populated in train, val, test)
