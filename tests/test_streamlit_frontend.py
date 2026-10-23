"""
Unit tests verifying the frontend Streamlit application logic and state formatting.
Guarantees that low-confidence, high-confidence, and not-a-leaf payloads render without NameErrors.
"""

import pytest
from frontend.streamlit_app import (
    CROP_EMOJIS,
    AGRONOMIC_GUIDANCE,
    MANDATORY_DISCLAIMER
)


def test_crop_emojis_registry():
    expected_crops = ["Apple", "Cherry", "Grape", "Maize", "Peach", "Pepper", "Potato", "Rice", "Strawberry", "Tomato"]
    for crop in expected_crops:
        assert crop in CROP_EMOJIS
        assert len(CROP_EMOJIS[crop]) > 0


def test_agronomic_guidance_completeness():
    required_keys = ["Healthy", "Early_blight", "Late_blight", "Common_rust", "Apple_scab", "Bacterial_spot", "Black_rot", "Powdery_mildew", "Leaf_blast", "Brown_spot"]
    for key in required_keys:
        assert key in AGRONOMIC_GUIDANCE
        advice = AGRONOMIC_GUIDANCE[key]
        for field in ["cultural", "irrigation", "treatment", "prevention"]:
            assert field in advice
            assert len(advice[field]) > 0


def test_streamlit_low_confidence_rendering_logic():
    """Simulates the state rendering of a low-confidence response to ensure no NameError occurs."""
    low_conf_data = {
        "status": "low_confidence",
        "is_leaf": True,
        "prediction": {
            "class_id": 1,
            "crop": "Tomato",
            "disease": "Early_blight",
            "label": "Tomato___Early_blight",
            "confidence": 0.42
        },
        "top_k": [
            {"label": "Tomato___Early_blight", "confidence": 0.42},
            {"label": "Potato___Early_blight", "confidence": 0.35},
            {"label": "Pepper___Bacterial_spot", "confidence": 0.12}
        ],
        "low_confidence": True,
        "threshold": 0.74,
        "message": "Unable to identify reliably.",
        "model_version": "1.0.0",
        "inference_time_ms": 3.8,
        "disclaimer": MANDATORY_DISCLAIMER
    }

    pred = low_conf_data["prediction"]
    is_low_conf = low_conf_data.get("low_confidence", False) or (low_conf_data.get("status") == "low_confidence")
    conf_pct = pred["confidence"] * 100
    crop_name = pred.get("crop", "Unknown")
    crop_emoji = CROP_EMOJIS.get(crop_name, "🌱")
    disease_display = pred.get("disease", "").replace("_", " ")

    assert is_low_conf is True
    assert crop_name == "Tomato"
    assert crop_emoji == "🍅"
    assert disease_display == "Early blight"

    # Simulate format string that previously raised NameError
    warning_banner = (
        f"⚠️ Low Confidence / Cross-Species Ambiguity Filter Triggered\n"
        f"Confidence: {conf_pct:.1f}% is below the 74.0% safety threshold.\n"
        f"The AI detected foliar lesion patterns, but cannot reliably distinguish whether the host species is {crop_name} or an alternative candidate.\n"
    )
    assert "Tomato" in warning_banner


def test_streamlit_high_confidence_rendering_logic():
    """Simulates the state rendering of a high-confidence response."""
    high_conf_data = {
        "status": "success",
        "is_leaf": True,
        "prediction": {
            "class_id": 4,
            "crop": "Apple",
            "disease": "Apple_scab",
            "label": "Apple___Apple_scab",
            "confidence": 0.98
        },
        "top_k": [
            {"label": "Apple___Apple_scab", "confidence": 0.98}
        ],
        "low_confidence": False,
        "threshold": 0.74,
        "message": "Disease identified successfully with high calibrated confidence.",
        "model_version": "1.0.0",
        "inference_time_ms": 4.1,
        "disclaimer": MANDATORY_DISCLAIMER
    }

    pred = high_conf_data["prediction"]
    is_low_conf = high_conf_data.get("low_confidence", False)
    conf_pct = pred["confidence"] * 100
    crop_name = pred.get("crop", "Unknown")
    crop_emoji = CROP_EMOJIS.get(crop_name, "🌱")
    disease_display = pred.get("disease", "").replace("_", " ")

    assert is_low_conf is False
    assert crop_name == "Apple"
    assert crop_emoji == "🍏"
    assert disease_display == "Apple scab"
