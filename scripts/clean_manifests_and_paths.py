"""
Clean and standardize all manifests:
1. Replaces hardcoded absolute machine paths with canonical relative paths.
2. Removes phantom overwritten duplicate rows in train.csv to ensure 100% 1-to-1 MD5 disk match.
3. Synchronizes master_group_split.csv and plantdoc_field_test_manifest.csv.
"""

import os
import hashlib
import pandas as pd

def clean_all_manifests():
    print("[*] Cleaning and standardizing manifests...")

    # 1. Train manifest
    train_df = pd.read_csv("data/manifests/train.csv")
    print(f"Initial train rows: {len(train_df)}")

    valid_train_indices = []
    for idx, r in train_df.iterrows():
        p = os.path.join("data", "processed", "train", str(r["class_label"]), str(r["filename"]))
        if os.path.exists(p):
            with open(p, "rb") as f:
                disk_md5 = hashlib.md5(f.read()).hexdigest()
            if disk_md5 == r["md5"]:
                valid_train_indices.append(idx)

    train_clean = train_df.loc[valid_train_indices].copy().reset_index(drop=True)
    print(f"Clean train rows (1-to-1 MD5 disk match): {len(train_clean)}")

    train_clean["filepath"] = train_clean.apply(
        lambda r: f"data/processed/train/{r['class_label']}/{r['filename']}", axis=1
    )

    # 2. Val manifest
    val_df = pd.read_csv("data/manifests/val.csv")
    print(f"Initial val rows: {len(val_df)}")
    val_df["filepath"] = val_df.apply(
        lambda r: f"data/processed/val/{r['class_label']}/{r['filename']}", axis=1
    )

    # 3. Test manifest
    test_df = pd.read_csv("data/manifests/test.csv")
    print(f"Initial test rows: {len(test_df)}")
    test_df["filepath"] = test_df.apply(
        lambda r: f"data/processed/test/{r['class_label']}/{r['filename']}", axis=1
    )

    # 4. Master group split
    master_clean = pd.concat([train_clean, val_df, test_df], ignore_index=True)

    # 5. PlantDoc manifest
    pdoc_df = pd.read_csv("data/manifests/plantdoc_field_test_manifest.csv")
    print(f"Initial PlantDoc rows: {len(pdoc_df)}")

    def clean_pdoc_path(p):
        norm = str(p).replace("\\", "/")
        marker = "data/plantdoc_field_test"
        if marker in norm:
            idx = norm.find(marker)
            norm = norm[idx:]
        if "plantdoc_why_are_my_pepper_plants_yellow_yellow_pepper_plants_yellow_leaves_green_veins_pepper_plants_yellow_veins_jpg.rf.c142a577c5389adf7fce038ecc363a28_box0.jpg" in norm:
            norm = "data/plantdoc_field_test/Pepper___Bacterial_spot/plantdoc_why_are_my_pepper_yellow_box0.jpg"
        elif "plantdoc_why_are_my_pepper_plants_yellow_yellow_pepper_plants_yellow_leaves_green_veins_pepper_plants_yellow_veins_jpg.rf.c142a577c5389adf7fce038ecc363a28_box1.jpg" in norm:
            norm = "data/plantdoc_field_test/Pepper___Bacterial_spot/plantdoc_why_are_my_pepper_yellow_box1.jpg"
        return norm

    pdoc_df["cropped_path"] = pdoc_df["cropped_path"].apply(clean_pdoc_path)

    # Verify all files exist
    checks = [
        ("train", train_clean, "filepath"),
        ("val", val_df, "filepath"),
        ("test", test_df, "filepath"),
        ("plantdoc", pdoc_df, "cropped_path")
    ]
    for name, df, col in checks:
        missing = [p for p in df[col] if not os.path.exists(p)]
        print(f"  [+] {name} verified: {len(df)} entries, missing on disk: {len(missing)}")
        assert len(missing) == 0, f"Found {len(missing)} missing files in {name}"

    # Save cleaned manifests
    train_clean.to_csv("data/manifests/train.csv", index=False)
    val_df.to_csv("data/manifests/val.csv", index=False)
    test_df.to_csv("data/manifests/test.csv", index=False)
    master_clean.to_csv("data/manifests/master_group_split.csv", index=False)
    pdoc_df.to_csv("data/manifests/plantdoc_field_test_manifest.csv", index=False)

    print("[+] Successfully updated and verified all manifests!")

if __name__ == "__main__":
    clean_all_manifests()
