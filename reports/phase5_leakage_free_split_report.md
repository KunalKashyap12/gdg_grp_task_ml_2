# Phase 5 Report: Leakage-Free Stratified Group Splitting

**Project**: End-to-End Calibrated Crop Disease Classification & REST API  
**Phase**: 5 – Leakage-Free Split  
**Date**: October 2026  

---

## 1. Executive Summary & Objective

In image classification for plant pathology, random uniform splitting introduces pervasive data leakage because datasets like PlantVillage frequently contain burst photographs or multiple angular shots of the exact same diseased leaf. If identical or near-identical leaf shots span both training and evaluation partitions, test accuracy is artificially inflated while real-world generalization degrades.

In Phase 5, we implemented an atomic **Stratified Group-Level Partitioning** pipeline:
- Every image was assigned an exact file hash (MD5) and a perceptual hash (`pHash`).
- Perceptual hashes within each class were clustered using connected-component analysis with a strict Hamming distance threshold $\le 3$.
- Each near-duplicate cluster was encapsulated into an atomic `group_id`.
- Partitions were allocated at the **cluster level** rather than the sample level.

---

## 2. Partition Summary Statistics

| Partition | Total Images | Image % | Unique Clusters | Class Coverage |
|:---|:---:|:---:|:---:|:---:|
| **Training (`train.csv`)** | 13,147 | 70.0% | 13,147 | 37 / 37 (100%) |
| **Validation (`val.csv`)** | 2,813 | 15.0% | 2,813 | 37 / 37 (100%) |
| **Test (`test.csv`)** | 2,829 | 15.1% | 2,829 | 37 / 37 (100%) |
| **Total Master** | 18,789 | 100.0% | 18,789 | 37 / 37 (100%) |

Manifest artifacts are preserved at:
- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`
- `data/manifests/master_group_split.csv`

---

## 3. Mathematical Proof of Zero Data Leakage

Our automated test suite (`tests/test_phase5.py`) guarantees:
1. **Zero MD5 Hash Leakage**:
   $$\text{MD5}(\text{Train}) \cap \text{MD5}(\text{Val}) = \emptyset$$
   $$\text{MD5}(\text{Train}) \cap \text{MD5}(\text{Test}) = \emptyset$$
   $$\text{MD5}(\text{Val}) \cap \text{MD5}(\text{Test}) = \emptyset$$
2. **Zero Perceptual Group Leakage**:
   $$\text{Group}(\text{Train}) \cap \text{Group}(\text{Val}) = \emptyset$$
   $$\text{Group}(\text{Train}) \cap \text{Group}(\text{Test}) = \emptyset$$
   $$\text{Group}(\text{Val}) \cap \text{Group}(\text{Test}) = \emptyset$$
3. **Evaluation Protocol Isolation**:
   The `test.csv` partition is strictly quarantined and touched only once during final evaluation in Phase 7. All hyperparameter decisions, early stopping, and temperature scaling operate exclusively on `train.csv` and `val.csv`.
