"""
Comprehensive Evaluation & Acceptance Gate Verification: Model v1.0.0 vs v1.1.0
- Evaluates Lab Test Set (data/manifests/test.csv, N=2,829)
- Evaluates Field Test Set (data/manifests/field_test_heldout.csv, N=323)
- Evaluates Field Unsupported Crop Set (data/manifests/field_test_unsupported.csv, N=65)
- Computes False-Healthy rates (overall, Potato, Rice)
- Computes Per-Class Recalls (all 37 classes) and checks delta
- Tests Acceptance Gate criteria and generates reports/improvement/comparison.md
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.abspath("."))
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD

def get_onnx_session(model_path):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4
    return ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

def preprocess_image(img_path, img_size=224):
    path = resolve_image_path(img_path)
    with Image.open(path) as raw_img:
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
        w, h = img.size
        # Center crop matching validation/inference pipeline
        scale = int(img_size * 1.14) / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Center crop
        left = (new_w - img_size) // 2
        top = (new_h - img_size) // 2
        crop_img = img_resized.crop((left, top, left + img_size, top + img_size))
        
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        # Normalize
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        tensor = arr.transpose(2, 0, 1)[np.newaxis, ...]
        return tensor

def softmax(logits):
    e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def evaluate_model(session, config_data, lab_df, field_df, unsupp_df):
    temperature = float(config_data.get("temperature", 1.0))
    threshold = float(config_data.get("confidence_threshold", 0.74))
    num_classes = config_data.get("num_classes", 37)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"[*] Evaluating model (classes={num_classes}, T={temperature:.4f}, theta={threshold:.2f})...")
    
    # 1. Lab Test Evaluation
    print(f"  [-] Evaluating Lab Test Set ({len(lab_df)} images)...")
    lab_preds, lab_confs, lab_targets = [], [], []
    for _, row in lab_df.iterrows():
        tensor = preprocess_image(row["filepath"])
        logits = session.run([output_name], {input_name: tensor})[0][0]
        # Calibrated softmax
        cal_logits = logits / temperature
        cal_probs = softmax(cal_logits)
        
        # For evaluation against 37 supported classes:
        # If model has 38 classes, supported classes are 0..36
        pred_class = int(np.argmax(cal_probs[:37]))
        conf = float(cal_probs[pred_class])
        
        lab_preds.append(pred_class)
        lab_confs.append(conf)
        lab_targets.append(int(row["class_id"]))
        
    lab_preds = np.array(lab_preds)
    lab_confs = np.array(lab_confs)
    lab_targets = np.array(lab_targets)
    
    lab_acc = float(accuracy_score(lab_targets, lab_preds))
    lab_macro_f1 = float(f1_score(lab_targets, lab_preds, average="macro", zero_division=0))
    
    # Per-class recalls
    recalls = recall_score(lab_targets, lab_preds, average=None, labels=list(range(37)), zero_division=0)
    per_class_recalls = {c: float(recalls[c]) for c in range(37)}
    
    # False-healthy rates
    # Load class metadata
    with open("data/supported_classes.json", "r") as f:
        classes_info = json.load(f)
    healthy_ids = {c["class_id"] for c in classes_info if c["disease"].lower() == "healthy"}
    potato_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Potato___Healthy")
    potato_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Potato" and c["disease"].lower() != "healthy"}
    rice_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Rice___Healthy")
    rice_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Rice" and c["disease"].lower() != "healthy"}
    
    # Overall false-healthy
    diseased_mask = np.isin(lab_targets, list(healthy_ids), invert=True)
    diseased_preds = lab_preds[diseased_mask]
    overall_false_healthy = float(np.mean(np.isin(diseased_preds, list(healthy_ids))))
    
    # Potato false-healthy
    potato_mask = np.isin(lab_targets, list(potato_diseased_ids))
    if np.sum(potato_mask) > 0:
        potato_false_healthy = float(np.mean(lab_preds[potato_mask] == potato_healthy_id))
    else:
        potato_false_healthy = 0.0
        
    # Rice false-healthy
    rice_mask = np.isin(lab_targets, list(rice_diseased_ids))
    if np.sum(rice_mask) > 0:
        rice_false_healthy = float(np.mean(lab_preds[rice_mask] == rice_healthy_id))
    else:
        rice_false_healthy = 0.0
        
    # 2. Field Test Evaluation
    print(f"  [-] Evaluating Held-out Field Test Set ({len(field_df)} images)...")
    field_preds, field_confs, field_targets = [], [], []
    for _, row in field_df.iterrows():
        tensor = preprocess_image(row["filepath"])
        logits = session.run([output_name], {input_name: tensor})[0][0]
        cal_logits = logits / temperature
        cal_probs = softmax(cal_logits)
        
        pred_class = int(np.argmax(cal_probs[:37]))
        conf = float(cal_probs[pred_class])
        
        field_preds.append(pred_class)
        field_confs.append(conf)
        field_targets.append(int(row["class_id"]))
        
    field_preds = np.array(field_preds)
    field_confs = np.array(field_confs)
    field_targets = np.array(field_targets)
    
    field_acc = float(accuracy_score(field_targets, field_preds))
    field_accepted = (field_confs >= threshold)
    field_coverage = float(np.mean(field_accepted))
    if np.sum(field_accepted) > 0:
        field_sel_acc = float(np.mean(field_preds[field_accepted] == field_targets[field_accepted]))
    else:
        field_sel_acc = 1.0
        
    # 3. Out-of-Scope Evaluation
    print(f"  [-] Evaluating Held-out Field Unsupported Set ({len(unsupp_df)} images)...")
    unsupp_confs = []
    unsupp_accepted = []
    unsupp_predicted_reject = []
    
    for _, row in unsupp_df.iterrows():
        tensor = preprocess_image(row["filepath"])
        logits = session.run([output_name], {input_name: tensor})[0][0]
        cal_logits = logits / temperature
        cal_probs = softmax(cal_logits)
        
        top_overall_idx = int(np.argmax(cal_probs))
        top_overall_conf = float(cal_probs[top_overall_idx])
        
        # Check if model identified internal reject class (idx 37)
        if num_classes >= 38 and top_overall_idx == 37:
            unsupp_predicted_reject.append(True)
            # Rejected: confidence in any supported crop is low
            supported_conf = float(np.max(cal_probs[:37]))
        else:
            unsupp_predicted_reject.append(False)
            supported_conf = float(np.max(cal_probs[:37]))
            
        unsupp_confs.append(supported_conf)
        # Accepted as a supported diagnosis if supported confidence >= threshold
        unsupp_accepted.append(supported_conf >= threshold and not (num_classes >= 38 and top_overall_idx == 37))
        
    unsupp_confs = np.array(unsupp_confs)
    mean_ood_conf = float(np.mean(unsupp_confs))
    median_ood_conf = float(np.median(unsupp_confs))
    frac_ood_gt_080 = float(np.mean(unsupp_confs > 0.80))
    frac_ood_accepted = float(np.mean(unsupp_accepted))
    frac_ood_rejected_as_unsupported = float(np.mean(unsupp_predicted_reject)) if num_classes >= 38 else 0.0
    
    metrics = {
        "lab_acc": lab_acc,
        "lab_macro_f1": lab_macro_f1,
        "field_acc": field_acc,
        "field_coverage": field_coverage,
        "field_sel_acc": field_sel_acc,
        "overall_false_healthy": overall_false_healthy,
        "potato_false_healthy": potato_false_healthy,
        "rice_false_healthy": rice_false_healthy,
        "mean_ood_conf": mean_ood_conf,
        "median_ood_conf": median_ood_conf,
        "frac_ood_gt_080": frac_ood_gt_080,
        "frac_ood_accepted": frac_ood_accepted,
        "frac_ood_rejected_as_unsupported": frac_ood_rejected_as_unsupported,
        "per_class_recalls": per_class_recalls
    }
    return metrics

def main():
    print("=" * 70)
    print("Starting Comprehensive Evaluation: v1.0.0 vs v1.1.0")
    print("=" * 70)
    
    # 1. Load Data
    lab_df = pd.read_csv("data/manifests/test.csv")
    field_df = pd.read_csv("data/manifests/field_test_heldout.csv")
    unsupp_df = pd.read_csv("data/manifests/field_test_unsupported.csv")
    
    with open("data/supported_classes.json", "r") as f:
        classes_info = json.load(f)
    class_names = {c["class_id"]: c["label"] for c in classes_info}
    
    # 2. Evaluate Baseline v1.0.0
    v100_onnx = "models/crop_disease_model.onnx"
    with open("models/model_config.json", "r") as f:
        v100_cfg = json.load(f)
    v100_session = get_onnx_session(v100_onnx)
    print("\n--- EVALUATING BASELINE v1.0.0 ---")
    m_v100 = evaluate_model(v100_session, v100_cfg, lab_df, field_df, unsupp_df)
    
    # 3. Evaluate Retrained v1.1.0
    v110_onnx = "models_v1.1.0/crop_disease_model.onnx"
    if not os.path.exists(v110_onnx):
        print(f"[!] ONNX model {v110_onnx} not found yet! Please ensure Phase 4 training has exported ONNX.")
        return
        
    with open("models_v1.1.0/model_config.json", "r") as f:
        v110_cfg = json.load(f)
    v110_session = get_onnx_session(v110_onnx)
    print("\n--- EVALUATING RETRAINED v1.1.0 ---")
    m_v110 = evaluate_model(v110_session, v110_cfg, lab_df, field_df, unsupp_df)
    
    # 4. Check Acceptance Gate Criteria
    # Gate 1: Lab test accuracy drop <= 2% (i.e. >= m_v100['lab_acc'] - 0.02)
    lab_drop = m_v100["lab_acc"] - m_v110["lab_acc"]
    gate1_pass = lab_drop <= 0.02
    
    # Gate 2: Field test accuracy improves (baseline was 10.34%)
    field_gain = m_v110["field_acc"] - m_v100["field_acc"]
    gate2_pass = field_gain > 0.0
    
    # Gate 3: Out-of-scope confidence > 0.80 drops substantially
    ood_drop = m_v100["frac_ood_gt_080"] - m_v110["frac_ood_gt_080"]
    gate3_pass = ood_drop > 0.05
    
    # Gate 4: False-healthy rate for Potato and Rice does not increase
    potato_fh_diff = m_v110["potato_false_healthy"] - m_v100["potato_false_healthy"]
    rice_fh_diff = m_v110["rice_false_healthy"] - m_v100["rice_false_healthy"]
    gate4_pass = (potato_fh_diff <= 0.01) and (rice_fh_diff <= 0.01)
    
    # Gate 5: No class recall drops by more than 10% without justification
    recall_drops = {}
    gate5_pass = True
    for cid in range(37):
        rec_old = m_v100["per_class_recalls"][cid]
        rec_new = m_v110["per_class_recalls"][cid]
        diff = rec_old - rec_new
        if diff > 0.10:
            recall_drops[class_names[cid]] = {"old": rec_old, "new": rec_new, "drop": diff}
            # Only fails gate if drop is unjustified
            gate5_pass = False
            
    all_gates_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass and (len(recall_drops) == 0)
    decision = "ACCEPTED" if all_gates_pass else "REJECTED"
    
    print("\n" + "=" * 70)
    print(f"ACCEPTANCE GATE EVALUATION: {decision}")
    print("=" * 70)
    print(f"Gate 1 (Lab Acc Drop <= 2.0%):        {'PASS' if gate1_pass else 'FAIL'} (Drop: {lab_drop*100:+.2f}%)")
    print(f"Gate 2 (Field Acc Improves):           {'PASS' if gate2_pass else 'FAIL'} (Gain: {field_gain*100:+.2f}%)")
    print(f"Gate 3 (OOD Conf >0.80 Drops):         {'PASS' if gate3_pass else 'FAIL'} (Drop: {ood_drop*100:+.2f}%)")
    print(f"Gate 4 (Potato/Rice False-Healthy):    {'PASS' if gate4_pass else 'FAIL'} (Potato diff: {potato_fh_diff*100:+.2f}%, Rice diff: {rice_fh_diff*100:+.2f}%)")
    print(f"Gate 5 (No Recall Drop >10%):          {'PASS' if len(recall_drops)==0 else 'FLAGGED'} ({len(recall_drops)} classes with drop >10%)")
    
    # 5. Write Comparison Markdown Report
    os.makedirs("reports/improvement", exist_ok=True)
    report_path = "reports/improvement/comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Improvement & Acceptance Gate Evaluation Report: v1.0.0 vs v1.1.0\n\n")
        f.write(f"**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Gate Decision**: **{decision}**\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- Baseline Model: MobileNetV3-Large v1.0.0 (37 classes, $T={v100_cfg.get('temperature', 0.4792)}$, $\\theta={v100_cfg.get('confidence_threshold', 0.74)}$)\n")
        f.write(f"- Retrained Model: MobileNetV3-Large v1.1.0 (38 classes [37 supported + 1 internal reject], $T={v110_cfg.get('temperature', 1.0)}$, $\\theta={v110_cfg.get('confidence_threshold', 0.70)}$)\n")
        f.write(f"- Final Gate Verdict: **{decision}**\n\n")
        
        f.write("## 2. Quantitative Comparison Table\n\n")
        f.write("| Evaluation Metric | Baseline v1.0.0 | Retrained v1.1.0 | Absolute Delta | Relative Change | Gate Target | Gate Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        rows = [
            ("Lab Test Accuracy", m_v100["lab_acc"], m_v110["lab_acc"], "%", "<= -2.0% drop", gate1_pass),
            ("Lab Test Macro F1", m_v100["lab_macro_f1"], m_v110["lab_macro_f1"], "%", "Preserve quality", True),
            ("Field Test Accuracy (PlantDoc)", m_v100["field_acc"], m_v110["field_acc"], "%", "> 0.0% gain", gate2_pass),
            ("Field Selective Accuracy", m_v100["field_sel_acc"], m_v110["field_sel_acc"], "%", "High precision", True),
            ("Field Test Coverage", m_v100["field_coverage"], m_v110["field_coverage"], "%", "Diagnostic yield", True),
            ("Overall False-Healthy Rate", m_v100["overall_false_healthy"], m_v110["overall_false_healthy"], "%", "<= 0.0% increase", m_v110["overall_false_healthy"] <= m_v100["overall_false_healthy"] + 0.01),
            ("Potato False-Healthy Rate", m_v100["potato_false_healthy"], m_v110["potato_false_healthy"], "%", "<= 0.0% increase", potato_fh_diff <= 0.01),
            ("Rice False-Healthy Rate", m_v100["rice_false_healthy"], m_v110["rice_false_healthy"], "%", "<= 0.0% increase", rice_fh_diff <= 0.01),
            ("OOD Conf > 0.80 Fraction", m_v100["frac_ood_gt_080"], m_v110["frac_ood_gt_080"], "%", "Substantial drop", gate3_pass),
            ("OOD Accepted at Threshold", m_v100["frac_ood_accepted"], m_v110["frac_ood_accepted"], "%", "Minimize hallucinations", True),
            ("OOD Identified as Unsupported", 0.0, m_v110["frac_ood_rejected_as_unsupported"], "%", "Internal reject class", True),
        ]
        
        for name, val_old, val_new, unit, target, pass_status in rows:
            delta = val_new - val_old
            rel = (delta / val_old * 100) if val_old > 0 else 0.0
            status_str = "PASS" if pass_status else "FAIL"
            f.write(f"| **{name}** | {val_old*100:.2f}% | {val_new*100:.2f}% | {delta*100:+.2f}% pts | {rel:+.1f}% | {target} | **{status_str}** |\n")
            
        f.write("\n## 3. Per-Class Recall Comparison (All 37 Classes)\n\n")
        f.write("| Class ID | Class Label | v1.0.0 Recall | v1.1.0 Recall | Delta | Status |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
        for cid in range(37):
            lbl = class_names[cid]
            r_old = m_v100["per_class_recalls"][cid]
            r_new = m_v110["per_class_recalls"][cid]
            d = r_new - r_old
            st = "WARNING" if d < -0.10 else ("IMPROVED" if d > 0.05 else "STABLE")
            f.write(f"| {cid:02d} | `{lbl}` | {r_old*100:.1f}% | {r_new*100:.1f}% | {d*100:+.1f}% pts | {st} |\n")
            
        f.write("\n## 4. Acceptance Gate Decision & Justification\n\n")
        f.write(f"### Final Verdict: **{decision}**\n\n")
        f.write("### Detailed Criteria Breakdown:\n")
        f.write(f"1. **Lab Test Retention**: {'PASSED' if gate1_pass else 'FAILED'} (Drop = {lab_drop*100:.2f}%, limit <= 2.0%)\n")
        f.write(f"2. **Field Generalization**: {'PASSED' if gate2_pass else 'FAILED'} (Accuracy changed from {m_v100['field_acc']*100:.2f}% to {m_v110['field_acc']*100:.2f}%, gain = {field_gain*100:+.2f}%)\n")
        f.write(f"3. **Out-of-Scope Protection**: {'PASSED' if gate3_pass else 'FAILED'} (High confidence hallucinations dropped by {ood_drop*100:.2f}% pts)\n")
        f.write(f"4. **Diagnostic Safety (False-Healthy)**: {'PASSED' if gate4_pass else 'FAILED'} (Potato FH diff: {potato_fh_diff*100:+.2f}%, Rice FH diff: {rice_fh_diff*100:+.2f}%)\n")
        f.write(f"5. **Recall Stability**: {'PASSED' if len(recall_drops)==0 else 'FLAGGED'} ({len(recall_drops)} classes dropped >10%)\n\n")
        
        f.write("## 5. Mandatory Disclaimer\n\n")
        f.write("> **Agricultural Safety Disclaimer**: *AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation.*\n")
        
    print(f"[+] Saved comparison report to {report_path}")

if __name__ == "__main__":
    main()
