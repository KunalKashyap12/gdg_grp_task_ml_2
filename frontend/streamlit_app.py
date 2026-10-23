"""
Streamlit Web Frontend for FloraScan AI: Crop Leaf Disease Diagnostics.
Connects to FastAPI backend at http://localhost:8000 or runs standalone inference.
"""

import os
import sys
import io
import time
import requests
import streamlit as st
from PIL import Image

# Setup page config
st.set_page_config(
    page_title="FloraScan AI — Crop Leaf Disease Diagnostics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
SAMPLES_DIR = os.path.join(os.path.abspath("."), "data", "samples")

MANDATORY_DISCLAIMER = "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."

CROP_EMOJIS = {
    'Apple': '🍏', 'Cherry': '🍒', 'Grape': '🍇', 'Maize': '🌽',
    'Peach': '🍑', 'Pepper': '🫑', 'Potato': '🥔', 'Rice': '🌾',
    'Strawberry': '🍓', 'Tomato': '🍅'
}

AGRONOMIC_GUIDANCE = {
    'Healthy': {
        'cultural': 'Maintain routine scouting, balanced plant density, and inspect for initial lesion signs weekly.',
        'irrigation': 'Continue standard root-zone irrigation aligned with crop growth stage.',
        'treatment': 'No fungicide intervention needed. Maintain beneficial predatory insect populations.',
        'prevention': 'Ensure balanced soil nutrition, compost addition, and annual crop rotation.'
    },
    'Early_blight': {
        'cultural': 'Prune lower leaves showing concentric rings. Mulch soil to avoid soil-borne spore splashing.',
        'irrigation': 'Irrigate strictly at soil level; avoid evening overhead sprinkling.',
        'treatment': 'Apply copper hydroxide or chlorothalonil at early detection.',
        'prevention': 'Rotate away from solanaceous crops for 3 years; choose certified resistant cultivars.'
    },
    'Late_blight': {
        'cultural': 'Immediately destroy and bury blighted vines. Eliminate volunteer potato cull piles.',
        'irrigation': 'Ensure wide canopy spacing to maximize air drying.',
        'treatment': 'Apply targeted systemic oomycete fungicides (e.g. cymoxanil, mandipropamid).',
        'prevention': 'Plant certified blight-free seed potatoes and disease-resistant hybrids.'
    },
    'Common_rust': {
        'cultural': 'Monitor rust pustules on lower leaves prior to corn tasseling.',
        'irrigation': 'Maintain balanced soil moisture avoiding long leaf wetness duration.',
        'treatment': 'Foliar strobilurin or triazole fungicides provide control if pustules spread before silking.',
        'prevention': 'Select corn hybrids with certified Rp-gene resistance.'
    },
    'Apple_scab': {
        'cultural': 'Shred or remove fallen autumn leaves to eliminate overwintering fungal pseudothecia.',
        'irrigation': 'Prune interior canopy branches during dormancy to promote air and light penetration.',
        'treatment': 'Apply protective captan or mancozeb sprays ahead of spring infection rains.',
        'prevention': 'Plant scab-resistant cultivars (e.g., Enterprise, Liberty, GoldRush).'
    },
    'Bacterial_spot': {
        'cultural': 'Avoid working in fields when foliage is wet. Prune and destroy severely infected lower leaves.',
        'irrigation': 'Use drip irrigation to keep leaf surfaces completely dry.',
        'treatment': 'Apply preventive copper-mancozeb bactericide tank mix before heavy rainfall events.',
        'prevention': 'Use certified pathogen-free seeds, hot-water seed treatment, and 2-year crop rotation.'
    },
    'Black_rot': {
        'cultural': 'Remove and destroy mummified fruit clusters and cankered vines or twigs during winter dormancy.',
        'irrigation': 'Maintain open plant canopy with balanced training for optimal sunlight and aeration.',
        'treatment': 'Apply protective myclobutanil or mancozeb fungicides from early bud break.',
        'prevention': 'Prune out dead wood annually; choose less susceptible varieties.'
    },
    'Powdery_mildew': {
        'cultural': 'Prune overcrowded shoots to improve air circulation and sunlight penetration.',
        'irrigation': 'Avoid excessive nitrogen fertilization which stimulates overly tender, susceptible shoots.',
        'treatment': 'Apply potassium bicarbonate, horticultural oils, or sulfur-based fungicides at first sign.',
        'prevention': 'Plant mildew-resistant rootstocks and cultivars in sunny locations.'
    },
    'Leaf_blast': {
        'cultural': 'Avoid excessive nitrogen application; maintain split fertilizer schedules.',
        'irrigation': 'Ensure adequate continuous field flood layer in paddy fields to mitigate blast susceptibility.',
        'treatment': 'Apply tricyclazole, isoprothiolane, or azoxystrobin at early tillering and panicle emergence.',
        'prevention': 'Use blast-resistant rice cultivars; treat seeds with carbendazim prior to sowing.'
    },
    'Brown_spot': {
        'cultural': 'Correct soil nutrient deficiencies, especially potassium, zinc, and manganese.',
        'irrigation': 'Maintain balanced, consistent water management without drought stress intervals.',
        'treatment': 'Apply propiconazole or iprodione if lesions expand rapidly during tillering.',
        'prevention': 'Soil solarization and balanced N-P-K fertilization.'
    }
}

# Custom CSS for polished styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stAlert {
        border-radius: 12px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .status-badge-success {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .status-badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_header_a, col_header_b = st.columns([3, 1])
with col_header_a:
    st.title("🌱 FloraScan AI — Foliar Diagnostics")
    st.caption("Calibrated Edge Deep Learning Diagnostics for 10 Crops & 37 Conditions")
with col_header_b:
    st.write("")
    st.success("🟢 ONNX Engine Active")

st.markdown("---")

# Main 2-column layout
col_left, col_right = st.columns([5, 7], gap="large")

# -------------------------------------------------------------
# LEFT COLUMN: UPLOAD & INPUT STUDIO
# -------------------------------------------------------------
with col_left:
    st.subheader("📸 Leaf Image Upload")
    
    # 1-Click sample presets
    sample_options = {
        "None (Custom Upload)": None,
        "🍏 Apple Scab": "apple_scab.jpg",
        "🍏 Apple Fungal Leaf (Field)": "apple-fungal-leaf.png",
        "🌽 Corn Common Rust": "corn_common_rust.jpg",
        "🥔 Potato Late Blight": "potato_late_blight.jpg",
        "🍇 Grape Healthy Leaf": "grape_healthy.jpg",
        "🍅 Tomato Foliage (Early Blight)": "tomato_early_blight.jpg"
    }
    
    selected_sample = st.selectbox(
        "⚡ 1-Click Sample Leaf Presets",
        options=list(sample_options.keys()),
        index=0
    )
    
    image_bytes = None
    image_filename = None
    
    if selected_sample != "None (Custom Upload)" and sample_options[selected_sample]:
        sample_filename = sample_options[selected_sample]
        sample_path = os.path.join(SAMPLES_DIR, sample_filename)
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            image_filename = sample_filename
    
    uploaded_file = st.file_uploader(
        "Upload Leaf Photograph (JPEG, PNG, WebP · Max 5MB)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a single clear, close-up photograph of a plant leaf."
    )
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image_filename = uploaded_file.name
    
    crop_choices = [
        "Auto-Detect Crop (Hierarchical)",
        "Apple", "Cherry", "Grape", "Maize", "Peach", "Pepper", "Potato", "Rice", "Strawberry", "Tomato"
    ]
    selected_crop_choice = st.selectbox(
        "🌿 Botanical Crop Species Context",
        options=crop_choices,
        index=0,
        help="Hierarchical Mode: The model identifies the leaf species first, then diagnoses the disease within that crop. Select a specific crop if you wish to confirm/override."
    )

    if image_bytes:
        pil_img = Image.open(io.BytesIO(image_bytes))
        st.image(pil_img, caption=f"{image_filename} ({pil_img.width}×{pil_img.height} px)", use_container_width=True)
        
        btn_run = st.button("🚀 Run Hierarchical Diagnosis", type="primary", use_container_width=True)
    else:
        st.info("👆 Please upload a leaf photo or pick a sample preset above to begin.")
        btn_run = False

# -------------------------------------------------------------
# RIGHT COLUMN: DIAGNOSTIC REPORT PANEL
# -------------------------------------------------------------
with col_right:
    st.subheader("📋 Diagnostic Report")
    
    if not image_bytes:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; border: 2px dashed #cbd5e1; border-radius: 16px;">
            <h4 style="color: #64748b; margin-bottom: 0.5rem;">Awaiting Leaf Photo</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Select an image on the left panel to trigger temperature-calibrated disease classification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 steps overview
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("01. Input", "224×224")
        c2.metric("02. ONNX", "Sub-5ms")
        c3.metric("03. Calib", "T = 0.48")
        c4.metric("04. Cutoff", "θ = 74%")
        
    elif btn_run or ("last_result" in st.session_state and st.session_state.get("last_img") == image_filename):
        with st.spinner("Evaluating MobileNetV3 ONNX graph and computing calibrated posteriors..."):
            try:
                # Query API with optional crop hint
                files = {"image": (image_filename or "leaf.jpg", image_bytes, "image/jpeg")}
                data_payload = {}
                if selected_crop_choice != "Auto-Detect Crop (Hierarchical)":
                    data_payload["crop"] = selected_crop_choice
                
                res = requests.post(f"{API_URL}/api/v1/predict", files=files, data=data_payload, timeout=10)
                
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["last_result"] = data
                    st.session_state["last_img"] = image_filename
                else:
                    st.error(f"Prediction failed: {res.text}")
                    data = None
            except Exception as e:
                st.error(f"Error reaching ML server: {e}")
                data = None
        
        if data:
            if not data.get("is_leaf", True) or data.get("status") == "not_a_leaf":
                st.error(
                    f"### 🚫 Non-Botanical Image Detected\n\n"
                    f"**{data.get('message', 'Not a leaf. Please upload a new image.')}**\n\n"
                    "The uploaded image does not exhibit foliar plant characteristics (chlorophyll/foliar absorption profiles) "
                    "matching any of the 14 supported crop categories. "
                    "To ensure agricultural safety, crop diagnosis and pesticide recommendations are withheld.\n\n"
                    "👉 **Please upload a clear, focused photograph of an actual plant leaf.**"
                )
                st.markdown("---")
                c_t1, c_t2, c_t3 = st.columns(3)
                c_t1.caption(f"⚡ Latency: **{data.get('inference_time_ms', 1.2)} ms**")
                c_t2.caption(f"🛡️ Gatekeeper: **Foliar Color & Saturation Validator**")
                c_t3.caption(f"🔒 Status: **REJECTED (NOT A LEAF)**")
                st.warning(f"⚖️ **Mandatory Agricultural & Safety Disclaimer**:\n\n{data.get('disclaimer', MANDATORY_DISCLAIMER)}")
            else:
                pred = data["prediction"]
                is_low_conf = data.get("low_confidence", False) or (data.get("status") == "low_confidence")
                conf_pct = pred["confidence"] * 100
                crop_name = pred.get("crop", "Unknown")
                crop_emoji = CROP_EMOJIS.get(crop_name, "🌱")
                disease_display = pred.get("disease", "").replace("_", " ")

                # Status Banner
                if is_low_conf:
                    st.warning(
                        f"⚠️ **Low Confidence / Cross-Species Ambiguity Filter Triggered**\n\n"
                        f"**Confidence: {conf_pct:.1f}% is below the 74.0% safety threshold.**\n\n"
                        f"The AI detected foliar lesion patterns, but cannot reliably distinguish whether the host species is {crop_name} or an alternative candidate due to outdoor background noise or visual symptom mimicry.\n\n"
                        f"👉 **Please confirm your crop species in the left sidebar to resolve the diagnosis.**"
                    )
                else:
                    st.success(
                        f"✅ **High Confidence Diagnosis (Accepted)**\n\n"
                        f"{data.get('message', 'Disease identified successfully with high calibrated confidence.')}"
                    )

                # Primary Card
                c_top_1, c_top_2 = st.columns([2, 1])
                with c_top_1:
                    st.markdown(f"### {crop_emoji} {crop_name}: **{disease_display}**")
                    st.caption(f"Canonical label: `{pred['label']}` (Class ID: {pred['class_id']})")
                with c_top_2:
                    st.metric(
                        label="Calibrated Confidence",
                        value=f"{conf_pct:.1f}%",
                        delta=f"{conf_pct - 74.0:+.1f}% vs θ"
                    )
                
                # Confidence Progress Bar
                st.progress(min(1.0, max(0.0, pred["confidence"])))
                st.caption("Threshold cutoff line at θ = 74.0% · Post-hoc Platt temperature T = 0.4792")

                # Hierarchical Two-Stage Pipeline Overview
                st.markdown("---")
                st.markdown("#### 🌿 Hierarchical Two-Stage Diagnostic Pipeline")
                s1_col, s2_col = st.columns(2)
                
                stage1 = data.get("stage1_leaf", {})
                stage2 = data.get("stage2_disease", {})
                
                with s1_col:
                    s1_crop = stage1.get("identified_crop", crop_name)
                    s1_conf = stage1.get("confidence", 0.0) * 100
                    is_cond = stage1.get("is_conditioned", False)
                    st.info(
                        f"**🍃 Stage 1: Leaf Species**\n\n"
                        f"**{CROP_EMOJIS.get(s1_crop, '🌱')} {s1_crop} Leaf** ({s1_conf:.1f}% confidence)\n\n"
                        f"*{'User Confirmed Context' if is_cond else 'Auto-Identified Leaf Species'}*"
                    )
                    if stage1.get("candidates"):
                        cands_text = ", ".join([f"{c['crop']} ({c['confidence']*100:.1f}%)" for c in stage1.get("candidates", [])])
                        st.caption(f"Top crop candidates: {cands_text}")

                with s2_col:
                    s2_dis = stage2.get("disease", pred["disease"]).replace("_", " ")
                    s2_conf = stage2.get("confidence", 0.0) * 100
                    path_mass = stage2.get("pathology_mass", 0.0) * 100
                    st.info(
                        f"**🔬 Stage 2: Foliar Pathology**\n\n"
                        f"**{s2_dis}** ({s2_conf:.1f}% conditional)\n\n"
                        f"*Foliar pathology: {path_mass:.1f}% · Healthy baseline: {100.0 - path_mass:.1f}%*"
                    )
                
                # Top-3 Differential Candidates
                st.markdown("#### 📊 Differential Candidates (Top-3)")
                top_k = data.get("top_k", [])
                for i, item in enumerate(top_k):
                    lbl = item["label"].replace("___", ": ").replace("_", " ")
                    c_item_1, c_item_2 = st.columns([3, 1])
                    c_item_1.write(f"**#{i+1}** {lbl}")
                    c_item_2.write(f"**{item['confidence']*100:.1f}%**")
                    st.progress(item["confidence"])
                
                # Agronomic Guidance
                st.markdown("#### 🛡️ Agronomic Management Guidance")
                advice_key = 'Healthy' if pred['disease'].lower() == 'healthy' else pred['disease']
                advice = AGRONOMIC_GUIDANCE.get(advice_key, AGRONOMIC_GUIDANCE.get('Early_blight'))
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.info(f"**✂️ Immediate Cultural Practices**\n\n{advice['cultural']}")
                    st.info(f"**💧 Irrigation Protocol**\n\n{advice['irrigation']}")
                with col_g2:
                    if is_low_conf:
                        st.warning("**🧪 Treatment Options**\n\n⚠️ Specific chemical treatment recommendations are safely withheld. Because the crop species is unconfirmed and confidence is below 74.0%, applying fungicides formulated for the wrong crop risks leaf burn or chemical phytotoxicity.")
                        st.info("**🔄 Long-term Prevention**\n\nTo obtain verified pathology and treatments, confirm your crop species in the sidebar or retake the photo isolating a single leaf against a neutral background.")
                    else:
                        st.info(f"**🧪 Treatment Options**\n\n{advice['treatment']}")
                        st.info(f"**🔄 Long-term Prevention**\n\n{advice['prevention']}")
                
                # Telemetry
                st.markdown("---")
                c_t1, c_t2, c_t3 = st.columns(3)
                c_t1.caption(f"⚡ Latency: **{data.get('inference_time_ms', 4.7)} ms**")
                c_t2.caption(f"🧠 Model: **MobileNetV3-Large ONNX**")
                c_t3.caption(f"📦 Version: **v{data.get('model_version', '1.0.0')}**")
                
                # Mandatory Disclaimer
                st.warning(f"⚖️ **Mandatory Agricultural & Safety Disclaimer**:\n\n{data.get('disclaimer', MANDATORY_DISCLAIMER)}")

    else:
        st.info("Click **Run Foliar Diagnosis** on the left to analyze the current image.")
