# Phase 3 Report: Class Balance & Weighting Strategy

## 1. Distribution Overview
- **Total Training Samples**: 13127 images across 37 classes.
- **Balancing Policy**: Raw classes with thousands of images were capped at 600 during ingestion to avoid majority class monopoly.
- **Imbalance Range**: Clean counts range from 138 (Rice Leaf blast) to 600 (PlantVillage classes). Imbalance ratio: ~4.3:1.
- **Reliability Audit**: Zero classes flagged as low-reliability (< 100 images). All 37 classes verified reliable.

## 2. Imbalance Mitigation Strategy
To ensure fair gradient contributions without introducing data leakage:
1. **No Pre-Split Oversampling**: Oversampling or duplicating before splitting was strictly avoided to prevent identical images across splits.
2. **Class-Weighted CrossEntropy Loss**: Loss penalty inversely proportional to class frequency:
   $$w_c = \frac{N}{K \cdot N_c}$$
3. **Smoothed Loss Weights**: Square-root smoothing option computed to mitigate gradient shock during backpropagation.

## 3. Class Counts & Computed Loss Weights

| Class ID | Label | Crop | Raw Count | Clean Count | Train Count | Loss Weight | Reliability |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | `Tomato___Bacterial_spot` | Tomato | 2127 | 600 | 420 | 0.845 | RELIABLE |
| 1 | `Tomato___Early_blight` | Tomato | 1000 | 600 | 420 | 0.845 | RELIABLE |
| 2 | `Tomato___Late_blight` | Tomato | 1909 | 600 | 420 | 0.845 | RELIABLE |
| 3 | `Tomato___Leaf_mold` | Tomato | 952 | 600 | 420 | 0.845 | RELIABLE |
| 4 | `Tomato___Septoria_leaf_spot` | Tomato | 1771 | 600 | 420 | 0.845 | RELIABLE |
| 5 | `Tomato___Spider_mites` | Tomato | 1676 | 600 | 420 | 0.845 | RELIABLE |
| 6 | `Tomato___Target_spot` | Tomato | 1404 | 600 | 420 | 0.845 | RELIABLE |
| 7 | `Tomato___Yellow_leaf_curl_virus` | Tomato | 5357 | 600 | 420 | 0.845 | RELIABLE |
| 8 | `Tomato___Mosaic_virus` | Tomato | 373 | 372 | 260 | 1.365 | RELIABLE |
| 9 | `Tomato___Healthy` | Tomato | 1591 | 600 | 420 | 0.845 | RELIABLE |
| 10 | `Potato___Early_blight` | Potato | 1000 | 600 | 420 | 0.845 | RELIABLE |
| 11 | `Potato___Late_blight` | Potato | 1000 | 600 | 420 | 0.845 | RELIABLE |
| 12 | `Potato___Healthy` | Potato | 152 | 152 | 106 | 3.347 | RELIABLE |
| 13 | `Pepper___Bacterial_spot` | Pepper | 997 | 600 | 420 | 0.845 | RELIABLE |
| 14 | `Pepper___Healthy` | Pepper | 1478 | 600 | 420 | 0.845 | RELIABLE |
| 15 | `Maize___Gray_leaf_spot` | Maize | 513 | 513 | 359 | 0.988 | RELIABLE |
| 16 | `Maize___Common_rust` | Maize | 1192 | 600 | 420 | 0.845 | RELIABLE |
| 17 | `Maize___Northern_leaf_blight` | Maize | 985 | 600 | 420 | 0.845 | RELIABLE |
| 18 | `Maize___Healthy` | Maize | 1162 | 600 | 420 | 0.845 | RELIABLE |
| 19 | `Apple___Apple_scab` | Apple | 630 | 600 | 420 | 0.845 | RELIABLE |
| 20 | `Apple___Black_rot` | Apple | 621 | 600 | 420 | 0.845 | RELIABLE |
| 21 | `Apple___Cedar_apple_rust` | Apple | 275 | 275 | 192 | 1.848 | RELIABLE |
| 22 | `Apple___Healthy` | Apple | 1645 | 600 | 420 | 0.845 | RELIABLE |
| 23 | `Grape___Black_rot` | Grape | 1180 | 600 | 420 | 0.845 | RELIABLE |
| 24 | `Grape___Esca_black_measles` | Grape | 1383 | 600 | 420 | 0.845 | RELIABLE |
| 25 | `Grape___Leaf_blight` | Grape | 1076 | 600 | 420 | 0.845 | RELIABLE |
| 26 | `Grape___Healthy` | Grape | 423 | 422 | 295 | 1.203 | RELIABLE |
| 27 | `Peach___Bacterial_spot` | Peach | 2297 | 600 | 420 | 0.845 | RELIABLE |
| 28 | `Peach___Healthy` | Peach | 360 | 360 | 251 | 1.413 | RELIABLE |
| 29 | `Strawberry___Leaf_scorch` | Strawberry | 1109 | 600 | 420 | 0.845 | RELIABLE |
| 30 | `Strawberry___Healthy` | Strawberry | 456 | 454 | 317 | 1.119 | RELIABLE |
| 31 | `Cherry___Powdery_mildew` | Cherry | 1052 | 600 | 420 | 0.845 | RELIABLE |
| 32 | `Cherry___Healthy` | Cherry | 854 | 600 | 420 | 0.845 | RELIABLE |
| 33 | `Rice___Bacterial_leaf_blight` | Rice | 237 | 184 | 128 | 2.772 | RELIABLE |
| 34 | `Rice___Brown_spot` | Rice | 250 | 190 | 133 | 2.668 | RELIABLE |
| 35 | `Rice___Leaf_blast` | Rice | 238 | 138 | 96 | 3.696 | RELIABLE |
| 36 | `Rice___Healthy` | Rice | 240 | 143 | 70 | 5.068 | RELIABLE |