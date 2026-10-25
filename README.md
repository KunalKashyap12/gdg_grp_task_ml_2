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

### Key Highlights:
- **10 Crops, 37 Classes**: Coverage across Tomato, Potato, Pepper (bell), Maize, Apple, Grape, Peach, Strawberry, Cherry, and Rice.
- **Zero Train/Serve Skew**: A unified preprocessing module ([`src/data/transforms.py`](src/data/transforms.py)) governs offline training, evaluation, and production serving.
- **Leakage-Free Group Splitting**: Perceptual hashing (`pHash`/`dHash`) clusters near-duplicate photos of identical leaves to guarantee zero cluster leakage across train (70%), validation (15%), and test (15%) splits.
- **Post-Hoc Confidence Calibration**: Platt temperature scaling ($T = 0.4792$) on validation logits reduces Expected Calibration Error (ECE) by **90.6%** (down to **1.95%** on the unseen test set).
- **Low-Confidence Cutoff ($\theta = 0.74$)**: Tuned on validation data to achieve **95.09% selective accuracy** on accepted test predictions, safely flagging uncertain and out-of-domain images as `"low_confidence"`.
- **Sub-5ms Inference**: Optimized ONNX Runtime engine achieves **4.68 ms/image** CPU latency (**4.96x speedup** over PyTorch).
- **Field Domain Shift Benchmarking**: Evaluated against the **PlantDoc** field dataset (3,996 images across 25 classes) documenting the real-world performance drop and verifying the effectiveness of the low-confidence reject filter.

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

### Final Model: MobileNetV3-Large (`models/best_model.pt`, `models/crop_disease_model.onnx`)

| Evaluation Dimension | Controlled Test Set (PlantVillage + Rice) | Outdoor Field Set (PlantDoc) | Notes |
| :--- | :--- | :--- | :--- |
| **Evaluated Classes** | 37 Classes (2,829 images) | 25 Mapped Classes (3,996 images) | Lab photography vs uncurated field photography |
| **Raw Top-1 Accuracy** | **88.41%** | 11.46% | Documents domain gap (-76.95% pts) |
| **Top-3 Accuracy** | **97.74%** | 24.10% | Correct diagnosis captured in top-3 candidates |
| **Crop Identification** | **96.29%** | 32.58% | Botanical crop species discrimination |
| **Disease Discrimination**| **89.36%** | - | Pathological condition discrimination |
| **Macro F1-Score** | **86.19%** | 7.44% | Unweighted mean over all 37 classes |
| **Calibrated ECE** | **1.95%** (was 20.81%) | - | **90.6% relative error reduction** ($T = 0.4792$) |
| **Selective Accuracy ($\theta=0.74$)**| **95.09%** (84.23% coverage) | **18.39%** (66.12% rejected) | Rejects uncertain field samples as `low_confidence` |
| **Inference CPU Latency** | **4.68 ms/sample** (~213 img/s) | 4.70 ms/sample | **4.96x faster** via ONNX Runtime vs PyTorch |

---

## 4. Repository Structure

```
gdg-grp-task-222/
├── api/                             # Production REST API (v1)
│   ├── main.py                      # FastAPI endpoints (/predict, /health, /classes, /model-info)
│   └── schemas.py                   # Pydantic v2 schemas and unified error envelopes
├── backend/                         # Spring Boot Java microservice client & controller
│   ├── pom.xml                      # Maven configuration with Spring Boot & Lombok
│   └── src/main/java/               # Java client, controller, DTOs & exception handling
├── data/                            # Registries & manifests
│   ├── manifests/                   # Leakage-free train.csv, val.csv, test.csv, plantdoc.csv
│   ├── supported_classes.json       # 37-class metadata registry
│   └── class_weights.json           # Inverse-frequency class weights
├── docs/                            # Cards, contracts & specifications
│   ├── API_CONTRACT.md              # Backend developer integration guide with curl examples
│   ├── BACKEND_INTEGRATION_GUIDE.md # End-to-end Spring Boot to ML service integration manual
│   ├── openapi.json                 # Auto-generated OpenAPI v3.1 schema
│   ├── MODEL_CARD.md                # Comprehensive Model Card
│   ├── DATASET_CARD.md              # Comprehensive Dataset Card
│   ├── LIMITATIONS.md               # 8 critical operational limitations & failure modes
│   └── PREPROCESSING_AUGMENTATION.md # Transforms & zero train/serve skew documentation
├── models/                          # Serialized model weights & configuration
│   ├── best_model.pt                # PyTorch MobileNetV3-Large checkpoint
│   ├── crop_disease_model.onnx      # Exported ONNX graph with dynamic batching (16.2 MB)
│   ├── calibration.json             # Temperature scaling metrics
│   └── model_config.json            # Serving parameters (threshold 0.74, temperature 0.4792)
├── notebooks/                       # Colab & Kaggle training notebook
│   └── crop_disease_model_training_colab.ipynb # Zero-upload GPU training notebook
├── reports/                         # Visual diagnostics and evaluation artifacts
│   ├── evaluation_report.md         # Full test set diagnostic report
│   ├── robustness_report.md         # Robustness, stress tests & out-of-scope report
│   ├── confusion_matrix.png         # Raw confusion matrix heatmap
│   ├── confusion_matrix_normalized.png # Normalized confusion matrix heatmap
│   ├── calibration_curves.png       # Reliability diagrams before and after calibration
│   ├── accuracy_vs_coverage.png     # Threshold risk-coverage tradeoff curve
│   ├── robustness_stress_chart.png  # 12 synthetic perturbation conditions bar chart
│   ├── gradcam_visualizations.png   # Grad-CAM foliar lesion attention overlays
│   ├── misclassified_gallery.png    # Diagnostic gallery of misclassified cases
│   └── test_metrics_per_class.csv   # Per-class precision, recall, F1, and support
├── src/                             # Core Python package
│   ├── data/                        # Ingestion, deduplication, transforms, split
│   ├── train/                       # Model definitions, two-stage fine-tuning
│   ├── eval/                        # Evaluation, Grad-CAM, PlantDoc benchmarking
│   ├── calibrate/                   # Temperature scaling & threshold selection
│   └── serve/                       # ONNX Runtime inference engine & export
├── tests/                           # Complete test suite
│   ├── test_api_v1.py               # API v1 endpoint integration tests
│   ├── test_phase8_robustness.py    # Robustness & stress test verification
│   ├── test_phase9_confidence.py    # Confidence calibration & threshold verification
│   ├── test_phase9.py               # ONNX export and parity tests
│   ├── test_phase7.py               # Evaluation deliverables tests
│   ├── test_phase5.py               # Leakage-free split tests
│   ├── test_phase4.py               # Preprocessing & augmentation tests
│   ├── test_phase3.py               # Class balance & weights tests
│   └── test_phase1.py               # Crop & class registry tests
├── Dockerfile                       # Multi-stage production container build
├── docker-compose.yml               # Container orchestration
├── requirements.txt                 # Pinned dependencies
├── config.yaml                      # Centralized configuration
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
