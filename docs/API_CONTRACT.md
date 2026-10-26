# API Contract & Backend Integration Guide (v1)

**Service**: Crop Leaf Disease Classification REST Service  
**Base URL**: `http://localhost:8000/api/v1` (or production host domain)  
**OpenAPI Specification**: [`docs/openapi.json`](openapi.json) | Interactive Swagger UI: `http://localhost:8000/docs`  
**Disclaimer**: *AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.*

---

## 1. Quick Integration Overview for Backend Engineers

This service provides single-leaf disease identification across **10 crops and 37 pathological classes** with **Platt-scaled calibrated confidences** and an automatic **low-confidence / out-of-distribution filter**.

### Standard Endpoints:
1. `POST /api/v1/predict` – Upload leaf photo (`multipart/form-data`, field: `"image"`).
2. `GET /api/v1/health` – Fast health check (`{"status": "ok", "model_loaded": true}`).
3. `GET /api/v1/classes` – Returns catalog of supported crops & diseases.
4. `GET /api/v1/model-info` – Returns model architecture, calibrated threshold, and metrics.

### Unified Error Structure:
All non-200 responses follow this consistent JSON shape:
```json
{
  "error": {
    "code": "BAD_REQUEST | FILE_TOO_LARGE | UNSUPPORTED_MEDIA_TYPE | VALIDATION_ERROR | INTERNAL_SERVER_ERROR",
    "message": "Human-readable explanation of error."
  }
}
```

---

## 2. Endpoints & Request/Response Contracts

### A. Predict Disease (`POST /api/v1/predict`)

- **Content-Type**: `multipart/form-data`
- **Field Name**: `image` (Required)
- **Supported File Types**: `.jpg`, `.jpeg`, `.png`, `.webp`
- **Payload Size Limit**: **5 MB** (enforced by server; rejects $>5$ MB with HTTP 413)

#### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -F "image=@/path/to/tomato_blight_leaf.jpg"
```

#### Example 1: Successful High-Confidence Diagnosis (`200 OK`)
When calibrated confidence $\ge \theta$ (e.g. $\theta = 0.70$):
```json
{
  "status": "success",
  "is_leaf": true,
  "prediction": {
    "class_id": 1,
    "crop": "Tomato",
    "disease": "Early_blight",
    "label": "Tomato___Early_blight",
    "confidence": 0.9124
  },
  "top_k": [
    {
      "label": "Tomato___Early_blight",
      "confidence": 0.9124
    },
    {
      "label": "Tomato___Septoria_leaf_spot",
      "confidence": 0.0612
    },
    {
      "label": "Tomato___Late_blight",
      "confidence": 0.0154
    }
  ],
  "low_confidence": false,
  "threshold": 0.70,
  "message": "Stage 1: Tomato leaf identified (98.4%). Stage 2: Early blight diagnosed (91.2% confidence, 98.5% foliar pathology).",
  "model_version": "1.1.0",
  "inference_time_ms": 4.68,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Example 2: Low-Confidence / Ambiguous Image (`200 OK`)
When calibrated confidence $< \theta$ or input exhibits low margin (<0.15) or high entropy (>1.50):
```json
{
  "status": "low_confidence",
  "is_leaf": true,
  "prediction": {
    "class_id": 10,
    "crop": "Potato",
    "disease": "Early_blight",
    "label": "Potato___Early_blight",
    "confidence": 0.5218
  },
  "top_k": [
    {
      "label": "Potato___Early_blight",
      "confidence": 0.5218
    },
    {
      "label": "Tomato___Early_blight",
      "confidence": 0.3142
    },
    {
      "label": "Potato___Late_blight",
      "confidence": 0.0984
    }
  ],
  "low_confidence": true,
  "threshold": 0.70,
  "message": "Unable to identify reliably. Retake a clear, well-lit, close-up photo of a single leaf.",
  "model_version": "1.1.0",
  "inference_time_ms": 4.62,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Example 3: Non-Leaf Gatekeeper Rejection (`200 OK`)
When the image is detected as non-foliar (wallpaper, graphic, skin/portrait, panoramic scenery, or blueprint):
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
  "threshold": 0.70,
  "message": "Not a leaf. Please upload a new image.",
  "model_version": "1.1.0",
  "inference_time_ms": 1.25,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Example 4: Unsupported Crop Variety (`200 OK`)
When the leaf belongs to a crop outside the 10 supported varieties (e.g. Squash, Blueberry, Soybean, Raspberry):
```json
{
  "status": "unsupported_crop",
  "is_leaf": true,
  "prediction": {
    "class_id": 37,
    "crop": "Unsupported",
    "disease": "Unsupported_crop",
    "label": "Unsupported___Unsupported_crop",
    "confidence": 0.8842
  },
  "top_k": [
    {"label": "Unsupported___Unsupported_crop", "confidence": 0.8842},
    {"label": "Grape___Leaf_blight", "confidence": 0.0412},
    {"label": "Tomato___Septoria_leaf_spot", "confidence": 0.0210}
  ],
  "low_confidence": true,
  "threshold": 0.70,
  "message": "This plant or crop is not currently supported by FloraScan. Supported crops: Apple, Cherry, Grape, Maize, Peach, Pepper, Potato, Rice, Strawberry, Tomato.",
  "model_version": "1.1.0",
  "inference_time_ms": 4.85,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

### Crop Name Normalization & Aliases:
The backend and ML service seamlessly support common agricultural parlance and colloquial aliases:
| User / Backend Input | Normalized Crop | Supported Status |
| :--- | :--- | :---: |
| `corn`, `Corn`, `maize`, `Maize` | `Maize` | Supported |
| `paddy`, `Paddy`, `rice`, `Rice` | `Rice` | Supported |
| `bell pepper`, `sweet pepper`, `pepper`, `Pepper` | `Pepper` | Supported |
| `Apple`, `Cherry`, `Grape`, `Peach`, `Potato`, `Strawberry`, `Tomato` | Preserved | Supported |
| Other crops (e.g. `Banana`, `Wheat`, `Soybean`) | N/A | HTTP 400 Bad Request or flagged as `unsupported_crop` |

### Recommended In-App Photography Guidance for Farmers:
To achieve the highest diagnostic precision and avoid `low_confidence` rejections:
1. **Single Leaf Focus**: Photograph one detached or isolated leaf flat against your palm or a neutral background.
2. **Close-Up Framing**: Fill at least 70% of the camera viewfinder with the affected leaf lamina.
3. **Diffused Daylight**: Avoid harsh direct noon sunlight glare or deep dark shadows; diffused morning or evening daylight produces the best color fidelity.
4. **Sharp Focus on Lesions**: Tap the screen to focus directly on visible spots, pustules, or discoloration.

#### Example Error Responses:

- **HTTP 400 Bad Request** (Corrupt or unreadable image):
  ```json
  {
    "error": {
      "code": "BAD_REQUEST",
      "message": "Uploaded file is corrupt or not a valid image format."
    }
  }
  ```

- **HTTP 413 Payload Too Large** (File exceeds 5 MB):
  ```json
  {
    "error": {
      "code": "FILE_TOO_LARGE",
      "message": "Uploaded image exceeds maximum allowed size of 5 MB (6.20 MB received)."
    }
  }
  ```

- **HTTP 415 Unsupported Media Type** (Invalid extension or non-image format):
  ```json
  {
    "error": {
      "code": "UNSUPPORTED_MEDIA_TYPE",
      "message": "Unsupported file format 'report.pdf'. Allowed formats: JPEG, PNG, WebP."
    }
  }
  ```

- **HTTP 422 Unprocessable Entity** (Missing form field `"image"`):
  ```json
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Missing or invalid field: Field required"
    }
  }
  ```

---

### B. Health Liveness Check (`GET /api/v1/health`)

#### cURL:
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```
#### Response (`200 OK`):
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### C. Supported Classes Registry (`GET /api/v1/classes`)

#### cURL:
```bash
curl -X GET "http://localhost:8000/api/v1/classes"
```
#### Response (`200 OK`):
```json
{
  "total_classes": 37,
  "total_crops": 10,
  "crops": [
    "Apple", "Cherry", "Grape", "Maize", "Peach",
    "Pepper", "Potato", "Rice", "Strawberry", "Tomato"
  ],
  "classes": [
    {
      "class_id": 0,
      "crop": "Tomato",
      "disease": "Bacterial_spot",
      "label": "Tomato___Bacterial_spot",
      "display_name": "Tomato: Bacterial Spot",
      "description": "Bacterial infection caused by Xanthomonas campestris pv. vesicatoria."
    }
  ],
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

---

### D. Model Info & Threshold Metadata (`GET /api/v1/model-info`)

#### cURL:
```bash
curl -X GET "http://localhost:8000/api/v1/model-info"
```
#### Response (`200 OK`):
```json
{
  "model_name": "mobilenet_v3_large",
  "model_version": "1.0.0",
  "input_size": [3, 224, 224],
  "num_classes": 37,
  "threshold": 0.74,
  "temperature": 0.4792,
  "metrics_summary": {
    "test_top1_accuracy": 0.8841,
    "test_selective_accuracy": 0.9509,
    "test_coverage_pct": 84.23,
    "latency_ms": 4.68
  },
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

---

## 3. Questions to Confirm with the Backend Developer

Before finalizing production deployment and mobile client integration, please review and confirm the following integration details:

1. **Form Field Name**:
   - Current: `"image"` (`multipart/form-data`).
   - *Confirmation needed*: Is `"image"` the standard naming convention used in your mobile upload service, or do you require `"file"` or `"photo"`?
2. **Authentication & API Keys**:
   - Current: Publicly accessible endpoints with CORS enabled.
   - *Confirmation needed*: Will authentication (e.g. Bearer JWT tokens, API Key in `X-API-Key` header) be handled upstream at your API Gateway (e.g. Kong, AWS API Gateway, NGINX), or should this service enforce auth?
3. **Base URL & Versioning**:
   - Current: `/api/v1/` prefix.
   - *Confirmation needed*: Does this match your microservice routing architecture?
4. **File Size Limit (5 MB)**:
   - Current: 5 MB payload cap.
   - *Confirmation needed*: Are uncompressed high-resolution mobile camera uploads pre-compressed client-side, or should the gateway accept up to 10 MB or 15 MB?
5. **Disease Treatment & Agronomic Advice Lookup**:
   - Current: The API returns the diagnosis, disease name, and plain-language symptom description.
   - *Confirmation needed*: Should this service connect to an internal database or return an advisory lookup (e.g. recommended organic/chemical remedies, spraying timings), or does the backend handle that logic separately?
6. **Telemetry & Privacy**:
   - Current: No image bytes or user photos are stored to disk (in-memory processing only for privacy).
   - *Confirmation needed*: Would you like an asynchronous opt-in toggle to archive low-confidence images into an S3 bucket for future active learning?
