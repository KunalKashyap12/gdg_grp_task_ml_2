"""
Pydantic Schemas for Phase 10 Crop Leaf Disease REST API (v1).
Matches the exact schema specifications with consistent error formatting.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


MANDATORY_DISCLAIMER = "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."


# -------------------------------------------------------------
# Consistent Error Shape: {"error": {"code": str, "message": str}}
# -------------------------------------------------------------
class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error explanation")


class ErrorResponse(BaseModel):
    error: ErrorDetail


# -------------------------------------------------------------
# Success Prediction Schemas (/api/v1/predict)
# -------------------------------------------------------------
class PredictionItem(BaseModel):
    class_id: int = Field(..., description="Zero-indexed class identifier")
    crop: str = Field(..., description="Identified crop name")
    disease: str = Field(..., description="Pathological condition or Healthy")
    label: str = Field(..., description="Canonical label (Crop___Disease)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated posterior confidence")


class TopKItem(BaseModel):
    label: str = Field(..., description="Candidate class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated posterior confidence")


class Stage1Leaf(BaseModel):
    identified_crop: str = Field(..., description="Stage 1 Botanical Leaf / Crop Species")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in botanical leaf species identification")
    is_conditioned: bool = Field(default=False, description="Whether crop was user-confirmed or auto-identified")
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked candidate crop species with probabilities")


class Stage2Disease(BaseModel):
    disease: str = Field(..., description="Stage 2 Pathological condition diagnosed within identified leaf")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in disease diagnosis conditional on leaf species")
    is_healthy: bool = Field(default=False, description="True if leaf shows healthy baseline morphology")
    pathology_mass: float = Field(default=0.0, description="Total foliar disease probability mass")
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked candidate diseases for this crop species")


class V1PredictionResponse(BaseModel):
    status: str = Field(..., description="'success' if confident, 'low_confidence' if below threshold, 'not_a_leaf' if non-leaf/wallpaper, 'unsupported_crop' if unsupported plant species")
    is_leaf: bool = Field(default=True, description="True if input is identified as a botanical plant leaf")
    prediction: PredictionItem = Field(..., description="Top diagnosis")
    top_k: List[TopKItem] = Field(..., min_length=1, description="Top-3 candidate predictions")
    low_confidence: bool = Field(..., description="True if confidence is below calibrated threshold")
    threshold: float = Field(..., description="Configured operating confidence threshold")
    message: str = Field(..., description="Diagnostic assessment or guidance message")
    model_version: str = Field(..., description="Serving model version string")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    stage1_leaf: Optional[Stage1Leaf] = Field(default=None, description="Stage 1: Botanical Leaf Species Identification")
    stage2_disease: Optional[Stage2Disease] = Field(default=None, description="Stage 2: Conditional Foliar Pathology Diagnosis")
    crop_conditioned: Optional[str] = Field(default=None, description="Botanical crop used for distribution conditioning")
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER, description="Mandatory agricultural disclaimer")


# -------------------------------------------------------------
# Service Health (/api/v1/health)
# -------------------------------------------------------------
class V1HealthResponse(BaseModel):
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(default=True)


# -------------------------------------------------------------
# Supported Classes Registry (/api/v1/classes)
# -------------------------------------------------------------
class ClassMetadata(BaseModel):
    class_id: int
    crop: str
    disease: str
    label: str
    display_name: str
    description: str


class V1ClassesResponse(BaseModel):
    total_classes: int
    total_crops: int
    crops: List[str]
    classes: List[ClassMetadata]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)


# -------------------------------------------------------------
# Model Info (/api/v1/model-info)
# -------------------------------------------------------------
class V1ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    input_size: List[int]
    num_classes: int
    threshold: float
    temperature: float
    metrics_summary: Dict[str, Any]
    disclaimer: str = Field(default=MANDATORY_DISCLAIMER)
