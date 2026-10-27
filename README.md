# Crop Leaf Disease Classification & Calibrated Decision Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CPU%20Sub--5ms-005CED.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v1%20Production%20REST-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

> [!IMPORTANT]
> **Mandatory Agricultural & Safety Disclaimer**:  
> **AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.**  
> Critical agricultural decisions, pesticide/fungicide applications, plant culling, or quarantine measures must always be validated by qualified agronomists, plant pathologists, or certified agricultural extension officers.

---

## 1. Project Overview

This repository provides an enterprise-ready, end-to-end machine learning system that diagnoses plant foliar diseases from a single leaf photo, provides a **temperature-calibrated confidence score** (reflecting true posterior probability), applies an **out-of-distribution low-confidence filter**, and serves predictions via a high-performance **FastAPI REST API (v1)** powered by an **ONNX Runtime** inference engine.

### Production Release (v1.1.0 - October 2026):
- **38 Output Classes**: 37 supported disease classes across 10 crops + 1 internal reject class (`Unsupported___Unsupported_crop`) protecting against out-of-scope leaf hallucinations.
- **Empirical Field Generalization**: Trained on combined lab + outdoor field photography (PlantDoc), lifting real-world outdoor field test accuracy from **15.48% to 47.68%** (+32.20% pts, >3x gain) and selective accuracy to **68.66%**.
- **Out-of-Distribution Protection**: High-confidence hallucinations on out-of-scope crops reduced from **27.69% to 0.00%** (-100% elimination), with 86.15% actively routed to `status="unsupported_crop"`.
- **Zero Train/Serve Skew**: A unified preprocessing module ([`src/data/transforms.py`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/src/data/transforms.py)) governs offline training, evaluation, and production serving.
- **Leakage-Free Group Splitting**: Perceptual hashing (`pHash`/`dHash`) and MD5 hash tracking guarantee zero near-duplicate or scene leakage across splits.
- **Post-Hoc Confidence Calibration**: Platt temperature scaling ($T = 0.6517$) reduces calibration error and aligns posterior probabilities.
- **Operational Low-Confidence Cutoff ($\theta = 0.70$)**: Yields **96.36% selective accuracy** on accepted predictions, routing uncertain samples to `"low_confidence"`.
- **Sub-5ms Inference**: Optimized ONNX Runtime engine achieves **4.68 ms/image** CPU latency.
- **Audit Certified**: Rigorously audited and certified for production in [`audit/POST_REMEDIATION_AUDIT_REPORT.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/audit/POST_REMEDIATION_AUDIT_REPORT.md).

---

## 2. Supported Crops & Classes Registry

Configured via `config.yaml` and cataloged in `data/supported_classes.json`:

| # | Botanical Crop | Supported Classes (Naming: `Crop___Disease`) |
| :-: | :--- | :--- |
| **1** | **Tomato** | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |
| **2** | **Potato** | Early blight, Late blight, Healthy |
| **3** | **Pepper (bell)** | Bacterial spot, Healthy |
| **4** | **Maize (Corn)** | Gray leaf spot, Common rust, Northern leaf blight, Healthy |
| **5** | **Apple** | Apple scab, Black rot, Cedar apple rust, Healthy |
| **6** | **Grape** | Black rot, Esca (black measles), Leaf blight, Healthy |
| **7** | **Peach** | Bacterial spot, Healthy |
| **8** | **Strawberry** | Leaf scorch, Healthy |
| **9** | **Cherry** | Powdery mildew, Healthy |
| **10**| **Rice (Paddy)** | Bacterial leaf blight, Brown spot, Leaf blast, Healthy *(Outdoor field dataset, GNU AGPL 3.0)* |

*Every crop contains a Healthy baseline class and at least one pathological disease class. All 37 classes have $\ge 100$ cleaned images.*

---

## 3. Benchmark & Model Performance

### Production Model v1.1.0 vs Baseline v1.0.0

| Evaluation Metric | Baseline v1.0.0 | Production v1.1.0 | Absolute Delta | Operational Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Lab Test Accuracy** | 89.86% | **96.43%** | +6.57% pts | Lab diagnostic accuracy enhanced |
| **Lab Macro F1-Score** | 87.77% | **94.97%** | +7.20% pts | Balanced multi-class quality |
| **Outdoor Field Accuracy (PlantDoc)** | 15.48% | **47.68%** | +32.20% pts | **>3x real-world field gain** |
| **Field Selective Accuracy** | 23.85% | **68.66%** | +44.80% pts | High reliability under field conditions |
| **Field Test Coverage** | 33.75% | **41.49%** | +7.74% pts | Higher actionable diagnostic throughput |
| **Overall False-Healthy Rate** | 1.96% | **0.47%** | -1.49% pts | 76% reduction in missed disease risk |
| **Potato False-Healthy Rate** | 4.44% | **0.56%** | -3.89% pts | 87.5% reduction in false-healthy |
| **Rice False-Healthy Rate** | 16.25% | **2.50%** | -13.75% pts | 84.6% reduction in false-healthy |
| **OOD Conf > 0.80 Rate** | 27.69% | **0.00%** | -27.69% pts | 100% elimination of confident hallucinations |
| **OOD Routed to Unsupported** | 0.00% | **86.15%** | +86.15% pts | Active rejection via 38th class |
| **Inference CPU Latency** | ~4.68 ms | **~4.72 ms** | +0.04 ms | Sub-5ms real-time throughput maintained |

*Full comparative breakdown available in [`reports/improvement/comparison.md`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/reports/improvement/comparison.md) and [`audit/remediation_comparison.csv`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task(ml)-2/audit/remediation_comparison.csv).*

---

## 4. Repository Structure

```
gdg-grp-task(ml)-2/
├── api/                             # Production REST API (v1)
│   ├── main.py                      # FastAPI endpoints (/predict, /health, /classes, /model-info)
│   └── schemas.py                   # Pydantic v2 schemas and unified error envelopes
├── audit/                           # Independent ML audit reports & verification artifacts
│   ├── AUDIT_REPORT.md              # Comprehensive initial & post-remediation audit report
│   ├── POST_REMEDIATION_AUDIT_REPORT.md # Standalone re-audit & certification report
│   ├── remediation_comparison.csv   # Machine-readable before/after benchmarks
│   ├── per_class_metrics_v1.1.0.csv # Per-class metrics for production model v1.1.0
│   └── confusion_matrix.png         # Full confusion matrix heatmap
├── backend/                         # Spring Boot Java microservice client & controller
│   ├── pom.xml                      # Maven configuration with Spring Boot & Lombok
│   └── src/main/java/               # Java client, controller, DTOs & exception handling
├── data/                            # Registries & manifests
│   ├── manifests/                   # Leakage-free train.csv, val.csv, test.csv, field splits
│   ├── supported_classes.json       # 37-class public metadata registry
│   └── class_weights.json           # Inverse-frequency class weights
├── docs/                            # Cards, contracts & specifications
│   ├── API_CONTRACT.md              # Backend developer integration guide with curl examples
│   ├── BACKEND_INTEGRATION_GUIDE.md # End-to-end Spring Boot to ML service integration manual
│   ├── openapi.json                 # Auto-generated OpenAPI v3.1 schema
│   ├── MODEL_CARD.md                # Comprehensive Model Card (v1.1.0 & v1.0.0)
│   ├── DATASET_CARD.md              # Comprehensive Dataset Card with leakage guarantees
│   ├── LIMITATIONS.md               # Operational limitations & mitigation strategies
│   └── PREPROCESSING_AUGMENTATION.md # Transforms & zero train/serve skew documentation
├── models_v1.1.0/                   # Production v1.1.0 model weights & calibration (38 classes)
│   ├── best_model.pt                # PyTorch MobileNetV3-Large checkpoint
│   ├── crop_disease_model.onnx      # Exported ONNX graph with dynamic batching (16.2 MB)
│   ├── calibration.json             # Platt temperature scaling parameters (T = 0.6517)
│   └── model_config.json            # Serving parameters (threshold 0.70)
├── models_backup_v1.0.0/            # Immutable fallback baseline model v1.0.0
├── reports/                         # Visual diagnostics and evaluation artifacts
│   ├── improvement/                 # Comparative before/after reports (v1.0.0 vs v1.1.0)
│   ├── evaluation_report.md         # Full test set diagnostic report
│   └── robustness_report.md         # Robustness, stress tests & out-of-scope report
├── src/                             # Core Python package
│   ├── data/                        # Ingestion, deduplication, transforms, split
│   ├── train/                       # Model definitions, two-stage fine-tuning
│   ├── eval/                        # Evaluation, Grad-CAM, PlantDoc benchmarking
│   ├── calibrate/                   # Temperature scaling & threshold selection
│   └── serve/                       # ONNX Runtime inference engine & export
├── tests/                           # Complete test suite (69 passing tests)
├── Dockerfile                       # Multi-stage production container build
├── docker-compose.yml               # Container orchestration
├── requirements.txt                 # Pinned dependencies
├── config.yaml                      # Centralized configuration (pointing to models_v1.1.0)
└── README.md
```

---

## 5. How to Reproduce (Setup, Train, Evaluate, Serve)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/example/crop-leaf-disease-classification.git
cd crop-leaf-disease-classification

# Create and activate virtual environment (Python 3.10+)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install pinned dependencies
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### 2. Data Preparation & Group Splitting
```bash
# Verify class registry and dataset integrity
python src/data/generate_supported_classes.py
python src/data/inspect_and_clean.py
python src/data/group_split.py
```

### 3. Model Training (PyTorch)
```bash
# Two-stage transfer learning with MobileNetV3-Large
python src/train/train.py
```
*(Or open [`notebooks/crop_disease_model_training_colab.ipynb`](notebooks/crop_disease_model_training_colab.ipynb) on Google Colab or Kaggle for cloud GPU training with zero local data upload required).*

### 4. Calibration & Threshold Optimization
```bash
# Post-hoc Temperature Scaling
python src/calibrate/temperature_scaling.py

# Low-confidence threshold selection on validation set
python src/calibrate/confidence_threshold.py
```

### 5. ONNX Export & Parity Verification
```bash
python src/serve/export_onnx.py
```

### 6. Full Evaluation & Robustness Stress Testing
```bash
python src/eval/evaluate.py
python src/eval/robustness_testing.py
```

### 7. Run Test Suite
```bash
python -m pytest tests/ -v
```

---

## 6. Running the Prediction API

### Option A: Local Uvicorn Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option B: Docker Container Deployment
```bash
docker compose up --build
```
Or with standalone Docker:
```bash
docker build -t crop-disease-api .
docker run -d -p 8000:8000 --name crop_disease_api crop-disease-api
```
Verify container health:
```bash
curl -f http://localhost:8000/api/v1/health
```

---

## 7. Prediction API Endpoints & Usage (v1)

### 1. Foliar Disease Prediction (`POST /api/v1/predict`)
- **Method**: `POST`
- **Field**: `image` (`multipart/form-data`)
- **Allowed Formats**: JPEG, PNG, WebP (Max 5 MB)

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -F "image=@sample_tomato_early_blight.jpg"
```

#### Success Response (`status: "success"`):
```json
{
  "status": "success",
  "prediction": {
    "class_id": 1,
    "crop": "Tomato",
    "disease": "Early_blight",
    "label": "Tomato___Early_blight",
    "confidence": 0.9124
  },
  "top_k": [
    {"label": "Tomato___Early_blight", "confidence": 0.9124},
    {"label": "Tomato___Septoria_leaf_spot", "confidence": 0.0612},
    {"label": "Tomato___Late_blight", "confidence": 0.0154}
  ],
  "low_confidence": false,
  "threshold": 0.74,
  "message": "Disease identified successfully with high calibrated confidence.",
  "model_version": "1.0.0",
  "inference_time_ms": 4.68,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Low-Confidence Response (`status: "low_confidence"`):
Triggered when confidence $< 0.74$ or when given an ambiguous, blurry leaf image:
```json
{
  "status": "low_confidence",
  "prediction": {
    "class_id": 10,
    "crop": "Potato",
    "disease": "Early_blight",
    "label": "Potato___Early_blight",
    "confidence": 0.5218
  },
  "top_k": [
    {"label": "Potato___Early_blight", "confidence": 0.5218},
    {"label": "Tomato___Early_blight", "confidence": 0.3142},
    {"label": "Potato___Late_blight", "confidence": 0.0984}
  ],
  "low_confidence": true,
  "threshold": 0.74,
  "message": "Unable to identify reliably. Retake a clear, well-lit, close-up photo of a single leaf.",
  "model_version": "1.0.0",
  "inference_time_ms": 4.62,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Non-Botanical Image Rejection (`status: "not_a_leaf"`):
Triggered when the uploaded image is a wallpaper, technical blueprint, sketch, or non-plant object:
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
  "top_k": [
    {"label": "None", "confidence": 0.0},
    {"label": "None", "confidence": 0.0},
    {"label": "None", "confidence": 0.0}
  ],
  "low_confidence": true,
  "threshold": 0.74,
  "message": "Not a leaf. Please upload a new image.",
  "model_version": "1.0.0",
  "inference_time_ms": 0.01,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Consistent Error Responses:
```json
{
  "error": {
    "code": "BAD_REQUEST | FILE_TOO_LARGE | UNSUPPORTED_MEDIA_TYPE | VALIDATION_ERROR | INTERNAL_SERVER_ERROR",
    "message": "Human-readable explanation of error."
  }
}
```

### 2. Service Liveness Check (`GET /api/v1/health`)
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 3. Supported Classes Registry (`GET /api/v1/classes`)
```bash
curl -X GET "http://localhost:8000/api/v1/classes"
```

### 4. Model Metadata & Metrics (`GET /api/v1/model-info`)
```bash
curl -X GET "http://localhost:8000/api/v1/model-info"
```

---

## 8. Final Deliverables Checklist

- [x] **Dataset + Preprocessing Pipeline**:
  - Ingestion, deduplication via perceptual hashing (`pHash`/`dHash`), cleaning log (`reports/cleaning_log.csv`).
  - Leakage-free group manifests (`data/manifests/train.csv`, `val.csv`, `test.csv`).
  - Shared canonical transform module ([`src/data/transforms.py`](src/data/transforms.py)) ensuring zero train/serve skew.
- [x] **Trained & Exported Models**:
  - Checkpoint: `models/best_model.pt` (MobileNetV3-Large).
  - Production ONNX graph: `models/crop_disease_model.onnx` (16.2 MB, dynamic batching, 4.68ms CPU latency).
  - Calibration: `models/calibration.json` ($T = 0.4792$).
  - Configuration: `models/model_config.json` (threshold $\theta = 0.74$).
  - Supported classes: `data/supported_classes.json` (37 classes across 10 crops).
- [x] **Evaluation & Diagnostics Reports**:
  - Held-out test evaluation report (`reports/evaluation_report.md`).
  - Robustness, synthetic stress tests & out-of-scope report (`reports/robustness_report.md`).
  - Visual diagnostics: confusion matrices, Grad-CAM lesion saliency maps, accuracy vs. coverage curves, stress test bar chart, misclassified gallery.
- [x] **Prediction API / Serving**:
  - FastAPI production application (`api/main.py`) with `/api/v1/predict`, `/health`, `/classes`, `/model-info`.
  - Consistent error envelopes and input validation.
  - Exported OpenAPI specification (`docs/openapi.json`).
  - Backend integration guide (`docs/API_CONTRACT.md`) with cURL requests and questions for backend engineers.
  - Multi-stage `Dockerfile` and `docker-compose.yml`.
  - Automated test suites (`tests/`).
- [x] **Model & System Documentation**:
  - `README.md`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`, `docs/LIMITATIONS.md`, `docs/PREPROCESSING_AUGMENTATION.md`.
  - Colab GPU training notebook: `notebooks/crop_disease_model_training_colab.ipynb`.
