"""
Phase 5: Leakage-Free Stratified Group Splitting.
Uses precomputed perceptual hashes from reports/dataset_inspection.csv.
1. Clusters near-duplicates where pHash Hamming distance <= 3 into atomic group_ids.
2. Performs Stratified Group 70/15/15 splitting per class.
3. Guarantees that near-duplicate clusters NEVER split across partitions.
4. Moves/organizes files in data/processed/train, val, test accordingly.
5. Saves manifests: data/manifests/train.csv, val.csv, test.csv with MD5, pHash, dHash, and group_id.
"""

import os
import shutil
import json
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import imagehash
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def cluster_hashes(phash_list, max_hamming_dist=3):
    """Clusters perceptual hashes using connected components."""
    n = len(phash_list)
    phash_objs = [imagehash.hex_to_hash(h) for h in phash_list]
    adj = defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            dist = phash_objs[i] - phash_objs[j]
            if dist <= max_hamming_dist:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    cluster_map = {}
    cluster_id = 0

    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            component = []
            while queue:
                curr = queue.pop()
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            for m in component:
                cluster_map[m] = cluster_id
            cluster_id += 1

    return cluster_map


def build_leakage_free_splits(seed=42):
    print("[*] Phase 5: Building leakage-free stratified group splits...")
    set_seed(seed)

    insp_csv = "reports/dataset_inspection.csv"
    if not os.path.exists(insp_csv):
        raise FileNotFoundError(f"{insp_csv} not found")

    df = pd.read_csv(insp_csv)
    print(f"[*] Ingested {len(df)} verified images across {df['class_label'].nunique()} classes.")

    with open("data/supported_classes.json", "r", encoding="utf-8") as f:
        supported_classes = json.load(f)
    class_to_id = {c["label"]: c["class_id"] for c in supported_classes}

    all_records = []
    global_group_offset = 0

    for class_label, group_df in tqdm(df.groupby("class_label"), desc="Clustering by class"):
        group_df = group_df.reset_index(drop=True)
        class_id = class_to_id[class_label]
        phash_list = group_df["phash"].tolist()

        cluster_map = cluster_hashes(phash_list, max_hamming_dist=3)
        n_clusters = max(cluster_map.values()) + 1 if cluster_map else 0

        # Map cluster -> row indices
        cluster_to_indices = defaultdict(list)
        for idx, cid in cluster_map.items():
            cluster_to_indices[cid + global_group_offset].append(idx)

        global_group_offset += n_clusters

        # 70/15/15 Group Split
        cluster_ids = list(cluster_to_indices.keys())
        random.shuffle(cluster_ids)

        total_class_imgs = len(group_df)
        target_train = int(total_class_imgs * 0.70)
        target_val = int(total_class_imgs * 0.15)

        train_clusters, val_clusters, test_clusters = [], [], []
        train_count, val_count = 0, 0

        for cid in cluster_ids:
            c_size = len(cluster_to_indices[cid])
            if train_count + c_size <= target_train:
                train_clusters.append(cid)
                train_count += c_size
            elif val_count + c_size <= target_val:
                val_clusters.append(cid)
                val_count += c_size
            else:
                test_clusters.append(cid)

        # Fallback to ensure every split has samples
        if not val_clusters and len(train_clusters) > 1:
            val_clusters.append(train_clusters.pop())
        if not test_clusters and len(train_clusters) > 1:
            test_clusters.append(train_clusters.pop())

        for split_name, c_list in [("train", train_clusters), ("val", val_clusters), ("test", test_clusters)]:
            for cid in c_list:
                for idx in cluster_to_indices[cid]:
                    row = group_df.iloc[idx]
                    all_records.append({
                        "filepath": row["filepath"],
                        "filename": row["filename"],
                        "class_label": class_label,
                        "class_id": class_id,
                        "crop": class_label.split("___")[0],
                        "disease": class_label.split("___")[1],
                        "group_id": f"grp_{cid:06d}",
                        "md5": row["md5"],
                        "phash": row["phash"],
                        "dhash": row["dhash"],
                        "width": row["width"],
                        "height": row["height"],
                        "split": split_name
                    })

    final_df = pd.DataFrame(all_records)
    print(f"\n[+] Total records after leakage-free group partition: {len(final_df)}")
    print(f"[+] Total distinct clusters / groups formed: {final_df['group_id'].nunique()}")

    # Update files in data/processed if split changed
    print("[*] Synchronizing physical files in data/processed according to new group splits...")
    target_paths = []
    for _, row in final_df.iterrows():
        curr_path = row["filepath"]
        new_split = row["split"]
        c_label = row["class_label"]
        fname = row["filename"]
        target_path = os.path.join("data", "processed", new_split, c_label, fname)

        if os.path.abspath(curr_path) != os.path.abspath(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if os.path.exists(curr_path):
                shutil.move(curr_path, target_path)
        target_paths.append(target_path)

    final_df["filepath"] = target_paths

    train_df = final_df[final_df["split"] == "train"].reset_index(drop=True)
    val_df = final_df[final_df["split"] == "val"].reset_index(drop=True)
    test_df = final_df[final_df["split"] == "test"].reset_index(drop=True)

    print(f"[+] Partition counts: Train={len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%), "
          f"Val={len(val_df)} ({len(val_df)/len(final_df)*100:.1f}%), "
          f"Test={len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")

    # Save manifests
    os.makedirs("data/manifests", exist_ok=True)
    train_df.to_csv("data/manifests/train.csv", index=False)
    val_df.to_csv("data/manifests/val.csv", index=False)
    test_df.to_csv("data/manifests/test.csv", index=False)
    final_df.to_csv("data/manifests/master_group_split.csv", index=False)

    print("[+] Successfully exported data/manifests/train.csv, val.csv, test.csv, master_group_split.csv")
    return final_df


if __name__ == "__main__":
    build_leakage_free_splits()
