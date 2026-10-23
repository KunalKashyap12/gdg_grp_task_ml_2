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
When calibrated confidence $\ge 0.74$:
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
  "threshold": 0.74,
  "message": "Disease identified successfully with high calibrated confidence.",
  "model_version": "1.0.0",
  "inference_time_ms": 4.68,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

#### Example 2: Low-Confidence / Ambiguous Image (`200 OK`)
When calibrated confidence $< 0.74$ or input exhibits high entropy (blurry, bad lighting, out-of-domain leaf, or unsupported crop):
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
  "threshold": 0.74,
  "message": "Unable to identify reliably. Retake a clear, well-lit, close-up photo of a single leaf.",
  "model_version": "1.0.0",
  "inference_time_ms": 4.62,
  "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
}
```

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
