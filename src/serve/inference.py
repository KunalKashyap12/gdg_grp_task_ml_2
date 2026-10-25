"""
Production Inference Engine with ONNX Runtime, Calibrated Confidence, and Hierarchical Two-Stage Classification.

Hierarchical Pipeline:
- Stage 1: Leaf / Botanical Species Identification (marginal posterior distribution over 10 crops)
- Stage 2: Foliar Pathology / Disease Diagnosis conditional on identified or confirmed leaf species
- Post-hoc Temperature Scaling confidence calibration (T = 0.4792)
- Low-confidence and Out-of-Distribution thresholding (theta = 0.74, margin = 0.15, entropy = 1.60)
- Predicts v1 JSON format: status ("success" | "low_confidence"), top_k (3 items), disclaimer
- Legacy predict() method for backward compatibility
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import time
import json
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

sys.path.insert(0, os.path.abspath("."))
from src.data.transforms import preprocess_image_for_inference

MANDATORY_DISCLAIMER = "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."
LOW_CONF_MESSAGE = "Unable to identify reliably. Retake a clear, well-lit, close-up photo of a single leaf."
SUCCESS_MESSAGE = "Disease identified successfully with high calibrated confidence."


class CropDiseaseInferenceEngine:
    """Production ONNX inference engine with calibrated confidences and hierarchical two-stage classification."""
    def __init__(self,
                 onnx_model_path="models/crop_disease_model.onnx",
                 classes_path="data/supported_classes.json",
                 config_path="models/model_config.json"):

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model missing: {onnx_model_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Supported classes JSON missing: {classes_path}")

        # 1. Load Classes
        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes_list = json.load(f)
        self.class_map = {c["class_id"]: c for c in self.classes_list}
        self.num_classes = len(self.classes_list)

        # Precompute crop to class mappings for Stage 1 & Stage 2 hierarchy
        self.crops = sorted(list(set(c["crop"] for c in self.classes_list)))
        self.crop_to_indices = {
            crop: [c["class_id"] for c in self.classes_list if c["crop"] == crop]
            for crop in self.crops
        }

        # 2. Load Model Config & Thresholds
        self.temperature = 0.4792
        self.threshold = 0.74
        self.margin_threshold = 0.15
        self.entropy_threshold = 1.60
        self.model_version = "1.0.0"
        self.config_data = {}

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config_data = json.load(f)
                self.temperature = float(self.config_data.get("temperature", 0.4792))
                self.threshold = float(self.config_data.get("confidence_threshold", 0.74))
                self.margin_threshold = float(self.config_data.get("margin_threshold", 0.15))
                self.entropy_threshold = float(self.config_data.get("entropy_threshold", 1.60))
                self.model_version = str(self.config_data.get("model_version", "1.0.0"))
            except Exception as e:
                print(f"[!] Warning loading model_config.json: {e}")

        # 3. Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def softmax(self, logits):
        """Numerically stable softmax."""
        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def predict_crop_species(self, logits):
        """
        Stage 1: Botanical Leaf / Crop Species Identification.
        Computes marginal posterior distribution across all supported botanical species:
        P(Crop = c | x) = sum_{k in Classes(c)} P(k | x)
        """
        probs = self.softmax(logits)
        crop_probs = {}
        for crop, indices in self.crop_to_indices.items():
            crop_probs[crop] = float(np.sum(probs[indices]))

        sorted_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)
        top_crop, top_crop_prob = sorted_crops[0]
        crop_candidates = [
            {"crop": c, "confidence": round(p, 4)}
            for c, p in sorted_crops[:3]
        ]
        return top_crop, top_crop_prob, crop_candidates, sorted_crops

    def validate_leaf_image(self, image_input):
        """
        Botanical Leaf Gatekeeper.
        Detects whether an image contains plant foliage or is a non-leaf (wallpaper, blueprint, screenshot, cartoon, object).
        Returns (is_leaf: bool, status: str, message: str)
        """
        try:
            if isinstance(image_input, (str, os.PathLike)):
                with Image.open(image_input) as img_raw:
                    img = ImageOps.exif_transpose(img_raw).convert("RGB")
            elif isinstance(image_input, bytes):
                import io
                with Image.open(io.BytesIO(image_input)) as img_raw:
                    img = ImageOps.exif_transpose(img_raw).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = ImageOps.exif_transpose(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                arr_in = image_input
                if np.issubdtype(arr_in.dtype, np.floating):
                    if arr_in.max() <= 1.0 and arr_in.min() >= 0.0:
                        arr_in = (arr_in * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr_in = arr_in.clip(0, 255).astype(np.uint8)
                else:
                    arr_in = arr_in.astype(np.uint8)

                if arr_in.ndim == 2:
                    img = Image.fromarray(np.stack([arr_in] * 3, axis=-1)).convert("RGB")
                elif arr_in.ndim == 3 and arr_in.shape[2] == 1:
                    img = Image.fromarray(np.repeat(arr_in, 3, axis=2)).convert("RGB")
                elif arr_in.ndim == 3 and arr_in.shape[2] == 4:
                    img = Image.fromarray(arr_in[:, :, :3]).convert("RGB")
                elif arr_in.ndim == 3 and arr_in.shape[2] == 3:
                    img = Image.fromarray(arr_in).convert("RGB")
                else:
                    return False, "not_a_leaf", "Not a leaf. Please upload a new image."
            else:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            arr = np.array(img)
            if arr.size == 0 or arr.ndim != 3:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            h_orig, w_orig = arr.shape[:2]
            aspect_ratio = max(w_orig / h_orig, h_orig / w_orig)

            # 1. Aspect Ratio Rejection:
            # Macro foliar photographs have standard photographic aspect ratios: 1:1, 4:3, 3:2, or 16:9.
            # Ratios > 2.05:1 (such as 21:9 ultrawide 2.37:1 desktop wallpapers or banners) or < 0.45:1
            # are characteristic of desktop wallpapers, web banners, or panoramic landscapes.
            if aspect_ratio > 2.05:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            # Downscale for ultra-fast gatekeeping (<1ms)
            if max(h_orig, w_orig) > 256:
                scale = 256.0 / max(h_orig, w_orig)
                new_w, new_h = max(1, int(w_orig * scale)), max(1, int(h_orig * scale))
                img_small = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
                small_arr = np.array(img_small)
            else:
                small_arr = arr

            # Solid blank / uniform color check (preserves low_confidence for blank gray image)
            if float(np.std(small_arr)) < 3.0:
                return False, "low_confidence", LOW_CONF_MESSAGE

            sh, sw = small_arr.shape[:2]
            r = small_arr[:, :, 0].astype(np.float32)
            g = small_arr[:, :, 1].astype(np.float32)
            b = small_arr[:, :, 2].astype(np.float32)

            # Excess Green index (ExG = 2G - R - B) standard in foliar ag-vision
            exg = 2.0 * g - r - b
            exg_fraction = float(np.mean(exg > 15.0))

            hsv = np.array(Image.fromarray(small_arr).convert("HSV"))
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # In PIL HSV, H is in [0, 255] corresponding to 0-360 degrees
            # Chlorophyll green: H ~ 35-115 (49 deg to 162 deg)
            # Chlorotic yellow: H ~ 22-35 (31 deg to 49 deg)
            # Necrotic brown / foliar lesion: H ~ 5-22 (7 deg to 31 deg)
            green_mask = (h >= 35) & (h <= 115) & (s >= 18) & (v >= 18)
            yellow_mask = (h >= 22) & (h < 35) & (s >= 20) & (v >= 20)
            necrotic_mask = (h >= 5) & (h < 22) & (s >= 20) & (v >= 15) & (v <= 215)

            green_fraction = float(np.mean(green_mask))
            yellow_fraction = float(np.mean(yellow_mask))
            vegetative_fraction = green_fraction + yellow_fraction
            low_sat_fraction = float(np.mean(s < 18))

            # Skin tone detection in YCbCr color space (detects human portraits / faces / selfies)
            cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
            cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
            skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173) & (s >= 20)
            skin_fraction = float(np.mean(skin_mask))

            # Dark background proportion (e.g. dark digital wallpapers, dark graphic frames)
            dark_mask = (v < 35)
            dark_fraction = float(np.mean(dark_mask))

            # Central region vegetative foliar presence (macro leaf subject test)
            ch1, ch2 = int(sh * 0.20), int(sh * 0.80)
            cw1, cw2 = int(sw * 0.20), int(sw * 0.80)
            center_veg = float(np.mean((green_mask | yellow_mask)[ch1:ch2, cw1:cw2]))

            # 1. Botanical Vegetation Anchor Test:
            # A genuine crop leaf MUST possess vegetative plant tissue (chlorophyll green or chlorotic yellow).
            # Wallpapers, posters, and non-foliar objects have negligible foliage.
            if vegetative_fraction < 0.18 and exg_fraction < 0.12:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            # 2. Human Portrait / Face / Selfie Filter:
            if skin_fraction > 0.08 and green_fraction < 0.05:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            # 3. Dark Wallpaper / Digital Graphic Rejection:
            if dark_fraction > 0.50 and vegetative_fraction < 0.15:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            # 4. Monochromatic / Blueprint / Technical Drawing Rejection:
            if low_sat_fraction > 0.65 and green_fraction < 0.08:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            # 5. Macro Foliar Leaf Subject Requirement:
            # When photographing a leaf for crop disease diagnosis, leaf foliage must occupy the center.
            # Scenic landscapes, distant horizons, or non-botanical scenery lack central foliar presence.
            if center_veg < 0.30 and vegetative_fraction < 0.35:
                return False, "not_a_leaf", "Not a leaf. Please upload a new image."

            return True, "valid", "Valid foliar leaf image"
        except Exception:
            return True, "valid", "Leaf validation bypass"

    def predict_v1(self, image_input, top_k=3, crop_hint=None, use_tta=True):
        """
        Runs calibrated hierarchical inference:
        - Stage 0: Botanical Leaf Gatekeeper (Filters out non-leaf / wallpaper / blueprints)
        - Stage 1: Leaf / Botanical Species Identification
        - Stage 2: Foliar Disease Diagnosis within identified leaf species
        - Multi-view Test-Time Augmentation (TTA) for enhanced stability
        - Formats according to Phase 10 API specification.
        """
        t0 = time.perf_counter()

        # 0. Botanical Leaf Gatekeeper
        is_leaf, gatekeeper_status, gatekeeper_msg = self.validate_leaf_image(image_input)
        if not is_leaf:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": gatekeeper_status,
                "is_leaf": False,
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
                "stage1_leaf": {
                    "identified_crop": "Unknown",
                    "confidence": 0.0,
                    "is_conditioned": False,
                    "candidates": []
                },
                "stage2_disease": {
                    "disease": "Not_a_leaf",
                    "confidence": 0.0,
                    "is_healthy": False,
                    "pathology_mass": 0.0,
                    "candidates": []
                },
                "low_confidence": True,
                "threshold": self.threshold,
                "crop_conditioned": None,
                "message": gatekeeper_msg,
                "model_version": self.model_version,
                "inference_time_ms": round(latency_ms, 2),
                "disclaimer": MANDATORY_DISCLAIMER
            }

        # 1. Preprocess with optional multi-view TTA
        input_tensor = preprocess_image_for_inference(
            image_input, img_size=224, return_tensor=False, use_tta=use_tta
        )
        if hasattr(input_tensor, "numpy"):
            input_tensor = input_tensor.numpy()
        input_tensor = input_tensor.astype(np.float32)

        # 2. ONNX Forward Pass
        ort_outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        if ort_outputs.shape[0] > 1:
            raw_logits = np.mean(ort_outputs, axis=0)
        else:
            raw_logits = ort_outputs[0]

        # Calibrated global probabilities
        calibrated_logits = raw_logits / self.temperature
        calibrated_probs = self.softmax(calibrated_logits)

        # 3. Check for User Crop Confirmation / Conditioning
        target_crop = None
        if crop_hint and isinstance(crop_hint, str):
            hint_clean = crop_hint.strip().lower()
            if hint_clean not in ["auto", "all", "none", "", "auto-detect", "auto_detect"]:
                for c in self.classes_list:
                    if c["crop"].lower() == hint_clean:
                        target_crop = c["crop"]
                        break

        # 4. Stage 1: Leaf / Botanical Crop Species Identification
        # Evaluated on calibrated distribution for mathematical coherence
        top_crop, top_crop_prob, crop_candidates, sorted_crops = self.predict_crop_species(calibrated_logits)
        crop_probs_map = dict(sorted_crops)

        # Active crop for Stage 2 (user confirmation overrides auto-detect)
        if target_crop:
            active_crop = target_crop
            is_conditioned = True
            stage1_conf = crop_probs_map.get(active_crop, 0.0)
            effective_threshold = 0.35
        else:
            active_crop = top_crop
            is_conditioned = False
            stage1_conf = top_crop_prob
            effective_threshold = self.threshold

        # 5. Stage 2: Foliar Pathology Diagnosis within active crop
        active_indices = self.crop_to_indices[active_crop]
        sub_logits = raw_logits[active_indices] / self.temperature
        sub_probs = self.softmax(sub_logits)

        sub_sorted_order = np.argsort(sub_probs)[::-1]
        best_sub_rank = sub_sorted_order[0]
        best_idx = active_indices[best_sub_rank]
        best_meta = self.class_map[best_idx]
        best_sub_prob = float(sub_probs[best_sub_rank])

        # Compute disease vs healthy breakdown within active crop
        healthy_prob = sum(
            float(sub_probs[i])
            for i, idx in enumerate(active_indices)
            if self.class_map[idx]["disease"].lower() == "healthy"
        )
        pathology_mass = max(0.0, 1.0 - healthy_prob)
        is_healthy = (best_meta["disease"].lower() == "healthy")

        # Top candidates for Stage 2 diseases
        crop_disease_candidates = []
        for rank_idx in sub_sorted_order:
            orig_idx = active_indices[rank_idx]
            meta = self.class_map[orig_idx]
            crop_disease_candidates.append({
                "disease": meta["disease"],
                "label": meta["label"],
                "confidence": round(float(sub_probs[rank_idx]), 4)
            })

        # Calculate joint calibrated confidence and global calibrated probs for OOD filtering
        entropy = -float(np.sum(calibrated_probs * np.log(calibrated_probs + 1e-12)))

        if is_conditioned:
            final_conf = best_sub_prob
            is_low_conf = (best_sub_prob < 0.35)
            # Top-k items strictly from conditioned crop in descending order
            top_k_count = min(max(1, top_k), len(active_indices))
            top_k_items = [
                {
                    "label": c["label"],
                    "confidence": c["confidence"]
                }
                for c in crop_disease_candidates[:top_k_count]
            ]
        else:
            final_conf = float(calibrated_probs[best_idx])
            sorted_cal = np.sort(calibrated_probs)[::-1]
            cal_margin = float(sorted_cal[0] - sorted_cal[1]) if len(sorted_cal) > 1 else 1.0

            # Raw uncalibrated metrics for out-of-distribution (OOD) protection
            unscaled_probs = self.softmax(raw_logits)
            raw_top_conf = float(np.max(unscaled_probs))
            raw_max_logit = float(np.max(raw_logits))

            is_low_conf = (
                (final_conf < self.threshold) or
                (entropy > self.entropy_threshold) or
                (cal_margin < self.margin_threshold) or
                (raw_top_conf < 0.25) or
                (raw_max_logit < 2.0)
            )
            # Top-k items from global calibrated distribution in strict descending order
            top_k_count = min(max(1, top_k), self.num_classes)
            top_k_items = []
            remaining_indices = np.argsort(calibrated_probs)[::-1]
            for idx in remaining_indices[:top_k_count]:
                meta = self.class_map[int(idx)]
                top_k_items.append({
                    "label": meta["label"],
                    "confidence": round(float(calibrated_probs[idx]), 4)
                })

        # Status and messages
        status = "low_confidence" if is_low_conf else "success"
        clean_dis = best_meta["disease"].replace("_", " ")

        if is_low_conf:
            message = LOW_CONF_MESSAGE
        else:
            if is_conditioned:
                if is_healthy:
                    message = f"Conditioned on {active_crop} foliage: Healthy plant baseline confirmed ({best_sub_prob*100:.1f}% confidence)."
                else:
                    message = f"Conditioned on {active_crop} foliage: {clean_dis} diagnosed ({best_sub_prob*100:.1f}% confidence, {pathology_mass*100:.1f}% total foliar pathology)."
            else:
                if is_healthy:
                    message = f"Stage 1: {active_crop} leaf identified ({stage1_conf*100:.1f}%). Stage 2: Healthy plant baseline confirmed ({best_sub_prob*100:.1f}% confidence)."
                else:
                    message = f"Stage 1: {active_crop} leaf identified ({stage1_conf*100:.1f}%). Stage 2: {clean_dis} diagnosed ({best_sub_prob*100:.1f}% confidence, {pathology_mass*100:.1f}% foliar pathology)."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": status,
            "is_leaf": True,
            "prediction": {
                "class_id": best_idx,
                "crop": best_meta["crop"],
                "disease": best_meta["disease"],
                "label": best_meta["label"],
                "confidence": round(final_conf, 4)
            },
            "top_k": top_k_items,
            "stage1_leaf": {
                "identified_crop": active_crop,
                "confidence": round(stage1_conf, 4),
                "is_conditioned": is_conditioned,
                "candidates": crop_candidates
            },
            "stage2_disease": {
                "disease": best_meta["disease"],
                "confidence": round(best_sub_prob, 4),
                "is_healthy": is_healthy,
                "pathology_mass": round(pathology_mass, 4),
                "candidates": crop_disease_candidates[:3]
            },
            "low_confidence": is_low_conf,
            "threshold": effective_threshold,
            "crop_conditioned": target_crop,
            "message": message,
            "model_version": self.model_version,
            "inference_time_ms": round(latency_ms, 2),
            "disclaimer": MANDATORY_DISCLAIMER
        }

    def predict(self, image_input, top_k=3):
        """
        Legacy predict method matching Phase 9 verification test contract.
        Includes Stage 0 Botanical Gatekeeper and calibrated confidence scoring.
        """
        t0 = time.perf_counter()
        is_leaf, gatekeeper_status, gatekeeper_msg = self.validate_leaf_image(image_input)
        if not is_leaf:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "is_leaf": False,
                "status": gatekeeper_status,
                "message": gatekeeper_msg,
                "prediction": {
                    "class_id": 0,
                    "crop": "Unknown",
                    "disease": "Not_a_leaf",
                    "label": "None",
                    "calibrated_confidence": 0.0
                },
                "top_candidates": [],
                "calibration": {
                    "is_calibrated": True,
                    "temperature": self.temperature
                },
                "performance": {
                    "latency_ms": round(latency_ms, 2)
                },
                "disclaimer": MANDATORY_DISCLAIMER
            }

        input_tensor = preprocess_image_for_inference(image_input, img_size=224, return_tensor=False)
        if hasattr(input_tensor, "numpy"):
            input_tensor = input_tensor.numpy()
        input_tensor = input_tensor.astype(np.float32)

        raw_logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]
        calibrated_logits = raw_logits / self.temperature
        calibrated_probs = self.softmax(calibrated_logits)

        effective_top_k = min(max(1, top_k), self.num_classes)
        top_k_indices = np.argsort(calibrated_probs)[::-1][:effective_top_k]
        best_idx = int(top_k_indices[0])
        best_meta = self.class_map[best_idx]
        best_conf = float(calibrated_probs[best_idx])

        top_candidates = []
        for rank, idx in enumerate(top_k_indices, 1):
            meta = self.class_map[int(idx)]
            top_candidates.append({
                "rank": rank,
                "class_id": int(idx),
                "crop": meta["crop"],
                "disease": meta["disease"],
                "label": meta["label"],
                "calibrated_confidence": round(float(calibrated_probs[idx]), 4)
            })

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "is_leaf": True,
            "prediction": {
                "class_id": best_idx,
                "crop": best_meta["crop"],
                "disease": best_meta["disease"],
                "label": best_meta["label"],
                "calibrated_confidence": round(best_conf, 4)
            },
            "top_candidates": top_candidates,
            "calibration": {
                "is_calibrated": True,
                "temperature": self.temperature
            },
            "performance": {
                "latency_ms": round(latency_ms, 2)
            },
            "disclaimer": MANDATORY_DISCLAIMER
        }


# Singleton instance accessor
_engine_instance = None


def get_inference_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CropDiseaseInferenceEngine()
    return _engine_instance


if __name__ == "__main__":
    engine = get_inference_engine()
    dummy = Image.new("RGB", (250, 250), color=(50, 150, 50))
    res = engine.predict_v1(dummy)
    print(json.dumps(res, indent=2))
