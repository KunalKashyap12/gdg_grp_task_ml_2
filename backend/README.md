# FloraScan AI — Spring Boot Backend Integration Guide

This directory contains the production-grade Java Spring Boot integration package to connect your Java backend with the Python FastAPI ML microservice.

---

## 📁 Package Structure

```
backend/
├── pom.xml                              # Maven build & dependency configuration
├── README.md                            # This integration guide
└── src/
    └── main/
        ├── java/com/florascan/backend/
        │   ├── config/
        │   │   ├── MlServiceConfig.java        # RestTemplate with timeout configuration
        │   │   └── MlServiceProperties.java    # Type-safe @ConfigurationProperties
        │   ├── controller/
        │   │   └── CropDiagnosisController.java# Exposes POST /api/crops/diagnose
        │   ├── dto/
        │   │   ├── CropPredictionResponse.java # Root diagnosis response
        │   │   ├── PredictionItem.java         # Top-1 diagnosis object
        │   │   ├── TopKItem.java               # Differential diagnosis candidate
        │   │   ├── Stage1Leaf.java             # Botanical species metadata
        │   │   ├── Stage2Disease.java          # Foliar pathology metadata
        │   │   ├── CandidateItem.java          # Ranked probability candidate
        │   │   ├── HealthResponse.java         # Healthcheck DTO
        │   │   └── ErrorResponse.java          # Standard error shape
        │   ├── exception/
        │   │   ├── GlobalExceptionHandler.java # Uniform REST error handler
        │   │   └── MlServiceException.java     # Custom ML communication exception
        │   └── service/
        │       └── MlServiceClient.java        # HTTP Client calling FastAPI /api/v1/predict
        └── resources/
            └── application.yml                 # Spring Boot YAML configuration
```

---

## 🚀 Quick Integration in 3 Steps

### Step 1: Copy Java Files
Copy the contents of `backend/src/main/java/com/florascan/backend/` into your Spring Boot project under `src/main/java/com/yourcompany/`.

### Step 2: Add Configuration to `application.yml`
Add the ML service connection properties:
```yaml
server:
  port: 8080

ml:
  service:
    url: ${ML_SERVICE_URL:http://localhost:8000}
    timeout-ms: 10000
    max-retries: 2
```

### Step 3: Run Both Services
1. **Start the ML Microservice** (Terminal 1):
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
2. **Start your Spring Boot Application** (Terminal 2):
   ```bash
   ./mvnw spring-boot:run
   ```

---

## 📡 Exposed Endpoints

### 1. Leaf Disease Diagnosis
* **URL**: `POST http://localhost:8080/api/crops/diagnose`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `file`: Leaf image file (`.jpg`, `.jpeg`, `.png`, `.webp`, max 5 MB) — *Required*
  * `crop_hint`: User crop selection, e.g. `"Tomato"`, `"Potato"`, `"Pepper"`, `"Corn"` / `"Maize"`, `"Paddy"` / `"Rice"` — *Optional (synonyms automatically mapped)*
  * `top_k`: Number of differential diagnoses (default: `3`) — *Optional*

#### Possible Status Values:
- `success`: Model classified the leaf with confidence above threshold.
- `low_confidence`: Leaf image identified, but confidence fell below the operational threshold ($\theta = 0.70$) or exhibited high diagnostic ambiguity.
- `not_a_leaf`: Non-foliar image, wallpaper, or noise detected and filtered by the botanical gatekeeper.
- `unsupported_crop`: The plant leaf belongs to an out-of-scope species not currently supported by FloraScan.

#### Example cURL Request:
```bash
curl -X POST "http://localhost:8080/api/crops/diagnose" \
     -F "file=@sample_leaf.jpg" \
     -F "crop_hint=corn" \
     -F "top_k=3"
```

#### Example JSON Response:
```json
{
  "status": "success",
  "is_leaf": true,
  "prediction": {
    "class_id": 16,
    "crop": "Corn",
    "disease": "Common rust",
    "label": "Maize___Common_rust",
    "confidence": 0.9252
  },
  "top_k": [
    { "label": "Maize___Common_rust", "confidence": 0.9252 },
    { "label": "Maize___Gray_leaf_spot", "confidence": 0.0418 },
    { "label": "Maize___Northern_leaf_blight", "confidence": 0.0210 }
  ],
  "low_confidence": false,
  "threshold": 0.70,
  "message": "Diagnosis complete: Common rust identified with 92.52% confidence.",
  "model_version": "1.1.0",
  "inference_time_ms": 32.5,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis."
}
```

### 2. ML Connectivity Health Check
* **URL**: `GET http://localhost:8080/api/crops/ml-health`
* **Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 37,
  "supported_crops_count": 10,
  "calibration_active": true,
  "temperature": 0.4792
}
```

---

## 🐳 Docker Compose Multi-Container Setup

To launch both the Spring Boot app and the Python ML service together:

```yaml
version: "3.8"

services:
  # Python ML Microservice
  ml-service:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: florascan_ml
    restart: unless-stopped
    ports:
      - "8000:8000"

  # Spring Boot Backend
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: florascan_backend
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - ML_SERVICE_URL=http://ml-service:8000
    depends_on:
      - ml-service
```
