"""
Production REST API for Crop Leaf Disease Classification (v1).

Serves:
- POST /api/v1/predict: Foliar classification with calibrated confidence & low-confidence thresholding
- GET  /api/v1/health: Liveness and model loading status
- GET  /api/v1/classes: Supported botanical crops & disease registry
- GET  /api/v1/model-info: Model architecture, version, input dimensions, thresholds, metrics

Error Format:
- Consistent shape: {"error": {"code": str, "message": str}}
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile, Form, Request, status
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, os.path.abspath("."))
from api.schemas import (
    V1PredictionResponse,
    V1HealthResponse,
    V1ClassesResponse,
    V1ModelInfoResponse,
    ClassMetadata,
    ErrorResponse,
    MANDATORY_DISCLAIMER
)
from src.serve.inference import get_inference_engine

# Setup request logger (does NOT store image bytes)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CropDiseaseAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ONNX inference engine and model config once at startup."""
    try:
        engine = get_inference_engine()
        logger.info(f"Inference engine initialized: {engine.model_version} (T={engine.temperature}, theta={engine.threshold})")
    except Exception as e:
        logger.error(f"Failed loading inference engine during startup: {e}")
    yield


app = FastAPI(
    title="Crop Leaf Disease Classification API",
    description=(
        "Production REST API for foliar crop disease classification with calibrated "
        "confidence scores and low-confidence threshold filtering across 10 agricultural crops.\n\n"
        "**Legal & Safety Disclaimer**: *AI-assisted indication only, not a guaranteed agricultural diagnosis. "
        "Consult a local agricultural expert for confirmation.*"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# Consistent Error Handlers: {"error": {"code": str, "message": str}}
# -------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    code_map = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        413: "FILE_TOO_LARGE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        422: "VALIDATION_ERROR",
        500: "INTERNAL_SERVER_ERROR"
    }
    error_code = code_map.get(exc.status_code, "ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": str(exc.detail),
            "error": {"code": error_code, "message": str(exc.detail)}
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)
    errors = exc.errors()
    msg = errors[0].get("msg", "Validation error") if errors else "Validation error"
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": f"Missing or invalid field: {msg}",
            "error": {"code": "VALIDATION_ERROR", "message": f"Missing or invalid field: {msg}"}
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error processing {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred during prediction processing.",
            "error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred during prediction processing."}
        }
    )


# -------------------------------------------------------------
# Root & Information
# -------------------------------------------------------------
@app.get("/", summary="Interactive Frontend & API Root Overview")
def root_endpoint(request: Request):
    """Serves the interactive web UI if accessed from a browser, or API JSON metadata otherwise."""
    accept = request.headers.get("accept", "").lower()
    html_file = os.path.join(os.path.abspath("."), "frontend", "index.html")
    if os.path.exists(html_file) and ("text/html" in accept and "application/json" not in accept):
        return FileResponse(html_file, media_type="text/html")
    return {
        "service": "Crop Leaf Disease Classification API",
        "version": "1.0.0",
        "documentation": "/docs",
        "web_ui": "/ui",
        "api_v1": {
            "predict": "/api/v1/predict",
            "health": "/api/v1/health",
            "classes": "/api/v1/classes",
            "model_info": "/api/v1/model-info",
            "samples": "/api/v1/samples"
        },
        "disclaimer": MANDATORY_DISCLAIMER
    }


@app.get("/health", summary="Legacy Service Health Check")
def legacy_health():
    """Legacy health check endpoint returning format expected by test suite."""
    engine = get_inference_engine()
    crops = set(c["crop"] for c in engine.classes_list)
    return {
        "status": "healthy",
        "model_loaded": engine.session is not None,
        "num_classes": engine.num_classes,
        "supported_crops_count": len(crops),
        "calibration_active": True,
        "temperature": engine.temperature,
        "disclaimer": MANDATORY_DISCLAIMER
    }


@app.get("/supported-classes", summary="Legacy Supported Classes Registry")
def legacy_supported_classes():
    """Legacy supported classes endpoint returning format expected by test suite."""
    engine = get_inference_engine()
    crops = sorted(list(set(c["crop"] for c in engine.classes_list)))
    return {
        "total_classes": len(engine.classes_list),
        "total_crops": len(crops),
        "crops": crops,
        "classes": engine.classes_list,
        "disclaimer": MANDATORY_DISCLAIMER
    }


@app.post("/predict", summary="Legacy Predict Endpoint")
async def legacy_predict(
    file: UploadFile = File(..., description="Crop leaf photo file (JPEG, PNG, WebP)"),
    top_k: int = 3
):
    """Legacy prediction endpoint accepting form field 'file'."""
    filename = (file.filename or "").lower()
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    if not filename.endswith(valid_exts):
        raise StarletteHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{filename}'. Allowed: {valid_exts}"
        )

    try:
        contents = await file.read()
    except Exception:
        raise StarletteHTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to read uploaded file.")

    if len(contents) == 0:
        raise StarletteHTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.verify()
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.load()
    except Exception:
        raise StarletteHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid or readable image."
        )

    engine = get_inference_engine()
    top_k_clamped = min(max(1, top_k), engine.num_classes)
    return engine.predict(contents, top_k=top_k_clamped)


@app.get("/ui", summary="Interactive Diagnostic Web UI", include_in_schema=False)
def get_ui():
    """Serves the FloraScan AI interactive web application."""
    html_file = os.path.join(os.path.abspath("."), "frontend", "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    return JSONResponse(status_code=404, content={"error": {"code": "NOT_FOUND", "message": "UI frontend file not found"}})


@app.get("/api/v1/samples", summary="List Test Sample Leaf Images")
def get_sample_leaves():
    """Returns catalog of preloaded photographic sample leaves for 1-click evaluation."""
    samples_dir = os.path.join(os.path.abspath("."), "data", "samples")
    if not os.path.exists(samples_dir):
        return []
    
    samples_meta = [
        {"filename": "apple-fungal-leaf.png", "crop": "Apple", "condition": "Apple Fungal Leaf (Scab/Black Rot)", "type": "Field Apple Leaf", "expected_status": "success"},
        {"filename": "apple_scab.jpg", "crop": "Apple", "condition": "Apple Scab", "type": "Fungal", "expected_status": "success"},
        {"filename": "corn_common_rust.jpg", "crop": "Maize", "condition": "Common Rust", "type": "Fungal", "expected_status": "success"},
        {"filename": "potato_late_blight.jpg", "crop": "Potato", "condition": "Late Blight", "type": "Oomycete", "expected_status": "success"},
        {"filename": "grape_healthy.jpg", "crop": "Grape", "condition": "Healthy Leaf", "type": "Healthy Baseline", "expected_status": "success"},
        {"filename": "tomato_early_blight.jpg", "crop": "Tomato", "condition": "Ambiguous Foliage", "type": "Low-Confidence Filter Demo", "expected_status": "low_confidence"},
        {"filename": "field_leaf.jpg", "crop": "Field Sample", "condition": "Outdoor Field Leaf", "type": "PlantDoc Field Benchmark", "expected_status": "success"},
        {"filename": "iron_man_wallpaper.jpg", "crop": "Non-Leaf", "condition": "Wallpaper/Graphic", "type": "Gatekeeper Rejection Demo", "expected_status": "not_a_leaf"}
    ]
    return [s for s in samples_meta if os.path.exists(os.path.join(samples_dir, s["filename"]))]


@app.get("/api/v1/samples/{filename}", summary="Get Sample Leaf Photo")
def get_sample_leaf(filename: str):
    """Serves raw image bytes of sample leaf for 1-click testing."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(os.path.abspath("."), "data", "samples", safe_name)
    if not os.path.exists(file_path):
        raise StarletteHTTPException(status_code=404, detail=f"Sample file '{safe_name}' not found.")
    lower_name = safe_name.lower()
    if lower_name.endswith(".png"):
        media_type = "image/png"
    elif lower_name.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


# -------------------------------------------------------------
# 1. GET /api/v1/health
# -------------------------------------------------------------
@app.get("/api/v1/health", response_model=V1HealthResponse, summary="Service & Model Liveness Check")
def get_health():
    """Returns {"status": "ok", "model_loaded": true}."""
    engine = get_inference_engine()
    model_loaded = (engine.session is not None)
    return {"status": "ok", "model_loaded": model_loaded}


# -------------------------------------------------------------
# 2. GET /api/v1/classes
# -------------------------------------------------------------
@app.get("/api/v1/classes", response_model=V1ClassesResponse, summary="List Supported Crops & Pathologies")
def get_classes():
    """Returns list of supported 10 crops and 37 disease/healthy classes."""
    engine = get_inference_engine()
    crops = sorted(list(set(c["crop"] for c in engine.classes_list)))
    class_items = [
        ClassMetadata(
            class_id=c["class_id"],
            crop=c["crop"],
            disease=c["disease"],
            label=c["label"],
            display_name=c["display_name"],
            description=c["description"]
        )
        for c in engine.classes_list
    ]
    return {
        "total_classes": len(class_items),
        "total_crops": len(crops),
        "crops": crops,
        "classes": class_items,
        "disclaimer": MANDATORY_DISCLAIMER
    }


# -------------------------------------------------------------
# 3. GET /api/v1/model-info
# -------------------------------------------------------------
@app.get("/api/v1/model-info", response_model=V1ModelInfoResponse, summary="Model Architecture & Metrics Metadata")
def get_model_info():
    """Returns model architecture, input size, threshold, temperature, and evaluation summary."""
    engine = get_inference_engine()
    cfg = engine.config_data
    return {
        "model_name": cfg.get("model_name", "mobilenet_v3_large"),
        "model_version": engine.model_version,
        "input_size": cfg.get("input_size", [3, 224, 224]),
        "num_classes": engine.num_classes,
        "threshold": engine.threshold,
        "temperature": engine.temperature,
        "metrics_summary": {
            "test_top1_accuracy": cfg.get("test_metrics", {}).get("raw_accuracy", 0.8841),
            "test_selective_accuracy": cfg.get("test_metrics", {}).get("selective_accuracy", 0.9509),
            "test_coverage_pct": cfg.get("test_metrics", {}).get("coverage", 0.8423) * 100,
            "latency_ms": 4.68
        },
        "disclaimer": MANDATORY_DISCLAIMER
    }


# -------------------------------------------------------------
# 4. POST /api/v1/predict
# -------------------------------------------------------------
@app.post(
    "/api/v1/predict",
    response_model=V1PredictionResponse,
    summary="Predict Crop Disease from Single Leaf Photograph",
    responses={
        200: {"description": "Diagnosis successful or low-confidence flagged", "model": V1PredictionResponse},
        400: {"description": "Corrupt or unreadable image file", "model": ErrorResponse},
        413: {"description": "File exceeds maximum 5 MB limit", "model": ErrorResponse},
        415: {"description": "Unsupported media format", "model": ErrorResponse},
        422: {"description": "Missing form field 'image'", "model": ErrorResponse}
    }
)
async def predict_foliar_disease(
    image: Optional[UploadFile] = File(None, description="Crop leaf photo file (JPEG, PNG, or WebP; maximum 5 MB)"),
    file: Optional[UploadFile] = File(None, description="Alternative field name for crop leaf photo"),
    crop: Optional[str] = Form(None, description="Optional botanical crop hint (e.g., Apple, Tomato, Potato, etc.)"),
    crop_hint: Optional[str] = Form(None, description="Alternative field name for botanical crop hint")
):
    """
    Submits a foliar image for calibrated disease identification.
    
    Accepts:
    - multipart/form-data with field 'image' (or 'file')
    - Formats: JPEG, PNG, WebP
    - Max size: 5 MB
    - Optional 'crop' (or 'crop_hint'): Target crop name to condition prediction distribution
    """
    t0 = time.time()
    MAX_BYTES = 5 * 1024 * 1024  # 5 MB

    target_file = image or file
    if target_file is None:
        raise StarletteHTTPException(
            status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
            detail="Missing required form field 'image'."
        )

    active_crop = crop or crop_hint

    # 1. Validate Media Extension & MIME Type
    filename = (target_file.filename or "").lower()
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    valid_mimes = ("image/jpeg", "image/png", "image/webp")
    has_valid_ext = any(filename.endswith(ext) for ext in valid_exts)
    has_valid_mime = (target_file.content_type in valid_mimes) if target_file.content_type else False

    if not has_valid_ext and not has_valid_mime:
        logger.warning(f"Rejected unsupported extension: {filename} (MIME: {target_file.content_type})")
        raise StarletteHTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{filename}'. Allowed formats: JPEG, PNG, WebP."
        )

    # 2. Read Bytes and Enforce 5 MB Limit
    try:
        contents = await target_file.read()
    except Exception as e:
        raise StarletteHTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to read uploaded file stream.")

    if len(contents) == 0:
        raise StarletteHTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    if len(contents) > MAX_BYTES:
        logger.warning(f"Rejected oversized payload: {len(contents)/(1024*1024):.2f} MB")
        raise StarletteHTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Uploaded image exceeds maximum allowed size of 5 MB ({len(contents)/(1024*1024):.2f} MB received)."
        )

    # 3. Verify Image Decodability with Pillow
    try:
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.verify()
        with Image.open(io.BytesIO(contents)) as pil_img:
            pil_img.load()
    except (UnidentifiedImageError, Exception) as e:
        logger.warning(f"Rejected corrupt image payload: {e}")
        raise StarletteHTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is corrupt or not a valid image format."
        )

    # 4. Validate Crop Hint if Provided
    engine = get_inference_engine()
    if active_crop and isinstance(active_crop, str):
        hint_clean = active_crop.strip().lower()
        if hint_clean not in ["auto", "all", "none", "", "auto-detect", "auto_detect"]:
            known_crops = [c.lower() for c in engine.crops]
            if hint_clean not in known_crops:
                raise StarletteHTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported crop hint '{active_crop}'. Supported crops: {', '.join(engine.crops)}"
                )

    # 5. Calibrated Prediction & Thresholding
    try:
        result = engine.predict_v1(contents, top_k=3, crop_hint=active_crop)
        elapsed_ms = (time.time() - t0) * 1000.0
        logger.info(f"Prediction: {result['prediction']['label']} ({result['status']}, {result['prediction']['confidence']*100:.1f}%) in {elapsed_ms:.1f}ms")
        return result
    except Exception as e:
        logger.error(f"Inference execution error: {e}")
        raise StarletteHTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference processing failed.")
