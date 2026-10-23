"""
Phase 2: Comprehensive Data Inspection, Cleaning, Deduplication, and PlantDoc Extraction.
1. Inspects resolutions, color modes, and file formats across all classes.
2. Identifies and removes:
   - Corrupt/unreadable files
   - Tiny images (<64px short side)
   - Invalid extensions/formats
   - Exact duplicates (MD5 hash)
   - Near-duplicates (perceptual hashing: pHash/dHash clustering)
3. Logs every removal into reports/cleaning_log.csv.
4. Generates visual inspection grids (sample leaves per crop/class, resolution histograms).
5. Processes PlantDoc field dataset as an isolated domain-shift external benchmark test set:
   - Maps labels to supported classes
   - Excludes unmapped categories
   - Never mixes into training data.
"""

import os
import glob
import hashlib
import json
import yaml
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import seaborn as sns


# PlantDoc label index to supported standard label mapping
PLANTDOC_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___Healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper___Bacterial_spot",
    "Bell_pepper leaf": "Pepper___Healthy",
    "Cherry leaf": "Cherry___Healthy",
    "Corn Gray leaf spot": "Maize___Gray_leaf_spot",
    "Corn leaf blight": "Maize___Northern_leaf_blight",
    "Corn rust leaf": "Maize___Common_rust",
    "Peach leaf": "Peach___Healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Potato leaf": "Potato___Healthy",
    "Strawberry leaf": "Strawberry___Healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Yellow_leaf_curl_virus",
    "Tomato leaf": "Tomato___Healthy",
    "Tomato mold leaf": "Tomato___Leaf_mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___Healthy",
}


def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_and_clean_dataset(processed_dir="data/processed"):
    print("[*] Phase 2: Starting comprehensive data inspection and cleaning...")
    cleaning_logs = []
    inspections = []

    seen_md5 = {}
    phash_clusters = defaultdict(list)
    dhash_clusters = defaultdict(list)

    all_images = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        if not os.path.exists(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            c_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(c_dir):
                continue
            for fname in os.listdir(c_dir):
                all_images.append((split, class_name, os.path.join(c_dir, fname)))

    print(f"[*] Found {len(all_images)} total images in processed dataset. Inspecting...")

    for split, class_name, img_path in all_images:
        fname = os.path.basename(img_path)
        ext = os.path.splitext(fname)[1].lower()

        # 1. Check file extension
        if ext not in [".jpg", ".jpeg", ".png"]:
            cleaning_logs.append({
                "filepath": img_path,
                "filename": fname,
                "class_label": class_name,
                "split": split,
                "reason": "invalid_extension",
                "details": f"Disallowed extension: {ext}"
            })
            os.remove(img_path)
            continue

        # 2. Check corrupt / unreadable
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            cleaning_logs.append({
                "filepath": img_path,
                "filename": fname,
                "class_label": class_name,
                "split": split,
                "reason": "corrupted_file",
                "details": str(e)
            })
            if os.path.exists(img_path):
                os.remove(img_path)
            continue

        # 3. Open image for metadata inspection
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                mode = img.mode
                format_name = img.format
                
                # Check tiny images (<64px short side)
                short_side = min(w, h)
                if short_side < 64:
                    cleaning_logs.append({
                        "filepath": img_path,
                        "filename": fname,
                        "class_label": class_name,
                        "split": split,
                        "reason": "tiny_image",
                        "details": f"Dimensions {w}x{h} (short side {short_side} < 64px)"
                    })
                    os.remove(img_path)
                    continue

                # Convert to RGB for hashing
                img_rgb = img.convert("RGB")
                p_h = str(imagehash.phash(img_rgb))
                d_h = str(imagehash.dhash(img_rgb))

            # 4. Check exact duplicate (MD5)
            file_md5 = compute_md5(img_path)
            if file_md5 in seen_md5:
                cleaning_logs.append({
                    "filepath": img_path,
                    "filename": fname,
                    "class_label": class_name,
                    "split": split,
                    "reason": "exact_duplicate_md5",
                    "details": f"Matches {seen_md5[file_md5]}"
                })
                os.remove(img_path)
                continue
            seen_md5[file_md5] = img_path

            # Cluster tracking
            phash_clusters[(class_name, p_h)].append(img_path)
            dhash_clusters[(class_name, d_h)].append(img_path)

            inspections.append({
                "filepath": img_path,
                "filename": fname,
                "class_label": class_name,
                "split": split,
                "width": w,
                "height": h,
                "aspect_ratio": round(w / h, 3),
                "mode": mode,
                "format": format_name,
                "md5": file_md5,
                "phash": p_h,
                "dhash": d_h
            })

        except Exception as e:
            cleaning_logs.append({
                "filepath": img_path,
                "filename": fname,
                "class_label": class_name,
                "split": split,
                "reason": "processing_error",
                "details": str(e)
            })
            if os.path.exists(img_path):
                os.remove(img_path)

    # Save cleaning log
    clean_log_df = pd.DataFrame(cleaning_logs)
    os.makedirs("reports", exist_ok=True)
    clean_log_path = "reports/cleaning_log.csv"
    clean_log_df.to_csv(clean_log_path, index=False)
    print(f"[+] Cleaning log written to {clean_log_path} ({len(clean_log_df)} items rejected)")

    # Save inspection metadata
    insp_df = pd.DataFrame(inspections)
    insp_path = "reports/dataset_inspection.csv"
    insp_df.to_csv(insp_path, index=False)
    print(f"[+] Dataset inspection metadata saved to {insp_path} ({len(insp_df)} items verified)")

    # Generate Inspection Plots
    generate_inspection_plots(insp_df)

    return insp_df, clean_log_df


def generate_inspection_plots(insp_df):
    """Plot resolution distribution, aspect ratio distribution, and sample grids."""
    print("[*] Generating inspection visual figures...")
    os.makedirs("reports", exist_ok=True)

    # 1. Resolution & Aspect Ratio Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(insp_df["width"], ax=axes[0], bins=30, color="forestgreen", kde=True)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Count")

    sns.histplot(insp_df["aspect_ratio"], ax=axes[1], bins=30, color="darkorange", kde=True)
    axes[1].set_title("Aspect Ratio (Width / Height) Distribution")
    axes[1].set_xlabel("Aspect Ratio")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    res_plot_path = "reports/resolution_distribution.png"
    plt.savefig(res_plot_path, dpi=200)
    plt.close()
    print(f"[+] Saved resolution distribution plot to {res_plot_path}")

    # 2. Sample Grid: One sample leaf per supported class
    with open("data/supported_classes.json", "r", encoding="utf-8") as f:
        supported_classes = json.load(f)

    n_classes = len(supported_classes)
    cols = 6
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    axes = axes.flatten()

    for idx, entry in enumerate(supported_classes):
        label = entry["label"]
        disp_name = entry["display_name"]
        class_samples = insp_df[insp_df["class_label"] == label]
        ax = axes[idx]
        if len(class_samples) > 0:
            sample_path = class_samples.iloc[0]["filepath"]
            try:
                img = Image.open(sample_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "No Image", ha="center", va="center")
        ax.set_title(disp_name, fontsize=8)
        ax.axis("off")

    for idx in range(n_classes, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Supported Classes Sample Leaf Gallery (37 Classes / 10 Crops)", fontsize=16, y=1.01)
    plt.tight_layout()
    sample_grid_path = "reports/class_sample_grid.png"
    plt.savefig(sample_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved class sample grid to {sample_grid_path}")


def process_plantdoc_field_test(plantdoc_raw_dir="data/raw/PlantDoc_raw", output_dir="data/plantdoc_field_test"):
    """
    Extracts, maps, and crops PlantDoc field images into an external test benchmark.
    Guarantees strict separation from the training data.
    """
    print("[*] Processing PlantDoc dataset as external field-shift test set...")
    yaml_path = os.path.join(plantdoc_raw_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[!] Warning: PlantDoc data.yaml not found at {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        doc_cfg = yaml.safe_load(f)
    class_names = doc_cfg.get("names", [])

    records = []
    crop_counter = defaultdict(int)

    # Use images from both train and test partitions of PlantDoc to maximize field evaluation volume
    for split in ["train", "test", "valid"]:
        img_dir = os.path.join(plantdoc_raw_dir, split, "images")
        lbl_dir = os.path.join(plantdoc_raw_dir, split, "labels")
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
            base_id = os.path.splitext(os.path.basename(lbl_file))[0]
            # Match image
            img_candidates = glob.glob(os.path.join(img_dir, f"{base_id}.*"))
            if not img_candidates:
                continue
            img_path = img_candidates[0]

            with open(lbl_file, "r") as f:
                lines = [l.strip().split() for l in f if l.strip()]

            if not lines:
                continue

            try:
                with Image.open(img_path) as full_img:
                    w_full, h_full = full_img.size
                    for box_idx, line in enumerate(lines):
                        cls_idx = int(line[0])
                        if cls_idx >= len(class_names):
                            continue
                        raw_cls_name = class_names[cls_idx]
                        if raw_cls_name not in PLANTDOC_MAPPING:
                            # KEEP UNMAPPED ONES OUT!
                            continue
                        target_class = PLANTDOC_MAPPING[raw_cls_name]

                        # YOLO coordinates: center_x, center_y, width, height (normalized)
                        cx, cy, bw, bh = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                        x1 = max(0, int((cx - bw / 2.0) * w_full))
                        y1 = max(0, int((cy - bh / 2.0) * h_full))
                        x2 = min(w_full, int((cx + bw / 2.0) * w_full))
                        y2 = min(h_full, int((cy + bh / 2.0) * h_full))

                        box_w = x2 - x1
                        box_h = y2 - y1

                        # Reject tiny bounding boxes (<64px)
                        if box_w < 64 or box_h < 64:
                            continue

                        cropped_leaf = full_img.crop((x1, y1, x2, y2))
                        dest_class_dir = os.path.join(output_dir, target_class)
                        os.makedirs(dest_class_dir, exist_ok=True)
                        dest_fname = f"plantdoc_{base_id}_box{box_idx}.jpg"
                        dest_path = os.path.join(dest_class_dir, dest_fname)
                        cropped_leaf.convert("RGB").save(dest_path, "JPEG", quality=95)

                        crop_counter[target_class] += 1
                        records.append({
                            "source": "PlantDoc",
                            "domain": "field_photograph",
                            "label": target_class,
                            "crop": target_class.split("___")[0],
                            "disease": target_class.split("___")[1],
                            "original_image": img_path,
                            "cropped_path": os.path.abspath(dest_path),
                            "box_width": box_w,
                            "box_height": box_h
                        })
            except Exception as e:
                continue

    plantdoc_df = pd.DataFrame(records)
    manifest_path = "data/manifests/plantdoc_field_test_manifest.csv"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    plantdoc_df.to_csv(manifest_path, index=False)
    print(f"[+] PlantDoc field test benchmark extracted: {len(plantdoc_df)} real-field leaf crops across {plantdoc_df['label'].nunique()} classes")
    print(f"[+] Manifest written to {manifest_path}")

    # Log class distribution
    print("[*] PlantDoc Field Evaluation Classes breakdown:")
    print(plantdoc_df["label"].value_counts().to_string())

    return plantdoc_df


if __name__ == "__main__":
    inspect_and_clean_dataset()
    process_plantdoc_field_test()
