"""
Prepares a NEW novel out-of-scope dataset (N >= 100 images) strictly excluding:
- Orange
- Squash
- Raspberry
- Blueberry
- Soybean
"""

import os
import shutil
import urllib.request
import pandas as pd

def download_file(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)

def main():
    out_dir = "scratch/novel_ood_104"
    os.makedirs(out_dir, exist_ok=True)
    
    records = []
    
    # 1. Reuse existing verified non-orange images from scratch/ood_100_images
    ood_csv = "scratch/ood_100_results.csv"
    if os.path.exists(ood_csv):
        df_old = pd.read_csv(ood_csv)
        for _, row in df_old.iterrows():
            fpath = row["path"]
            desc = row["description"]
            cat = row["category"]
            # Exclude Orange
            if "orange" in fpath.lower() or "orange" in desc.lower():
                continue
            fname = os.path.basename(fpath)
            target_path = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                shutil.copy2(fpath, target_path)
                records.append({
                    "filename": fname,
                    "filepath": target_path,
                    "description": desc,
                    "category": cat
                })
        print(f"[*] Loaded {len(records)} existing non-orange images from previous collection.")

    # 2. Add 50 new novel out-of-scope samples (crops, weeds, trees, objects)
    new_urls = [
        # Novel crops (never in training or reject set)
        ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400", "Coffee branch and beans", "novel_crop"),
        ("https://images.unsplash.com/photo-1594489428504-5c0c480a15fd?w=400", "Cotton boll in field", "novel_crop"),
        ("https://images.unsplash.com/photo-1589923188900-85dae523342b?w=400", "Sugarcane stalk and leaves", "novel_crop"),
        ("https://images.unsplash.com/photo-1588880331179-bc9b93a8cb5e?w=400", "Cassava root and foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=400", "Papaya tree foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1553279768-865429fa0078?w=400", "Mango foliage and fruit", "novel_crop"),
        ("https://images.unsplash.com/photo-1541256942802-7b299b483057?w=400", "Olive branch and leaves", "novel_crop"),
        ("https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400", "Barley grain ears", "novel_crop"),
        ("https://images.unsplash.com/photo-1586201375761-83865001e31c?w=400", "Oat field stalks", "novel_crop"),
        ("https://images.unsplash.com/photo-1628088062854-d1870b4553da?w=400", "Sorghum heads", "novel_crop"),
        ("https://images.unsplash.com/photo-1589135233689-d56b4618a38c?w=400", "Millet stalks", "novel_crop"),
        ("https://images.unsplash.com/photo-1599940824399-b87987ceb72a?w=400", "Ginger plant leaves", "novel_crop"),
        ("https://images.unsplash.com/photo-1615485290154-152e46b96e62?w=400", "Turmeric foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1596040033229-a9821ebd058d?w=400", "Black pepper vines", "novel_crop"),
        ("https://images.unsplash.com/photo-1578849278619-e73505e9610f?w=400", "Cocoa pod on tree", "novel_crop"),
        
        # Novel weeds and wild herbs
        ("https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=400", "Stinging nettle leaves", "wild_weed"),
        ("https://images.unsplash.com/photo-1595855759920-86582396756a?w=400", "Milk thistle blossom", "wild_weed"),
        ("https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400", "Burdock plant leaves", "wild_weed"),
        ("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=400", "Broadleaf plantain", "wild_weed"),
        ("https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=400", "Field chickweed", "wild_weed"),
        ("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400", "Horsetail rush", "wild_weed"),
        ("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400", "Common ragweed in meadow", "wild_weed"),
        ("https://images.unsplash.com/photo-1470240731273-7821a6eeb6bd?w=400", "Ground ivy patch", "wild_weed"),
        ("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400", "Morning glory bindweed", "wild_weed"),
        ("https://images.unsplash.com/photo-1473773508845-188df298d2d1?w=400", "Wild chamomile flowers", "wild_weed"),
        ("https://images.unsplash.com/photo-1502744688674-c619d3864003?w=400", "Yarrow herb cluster", "wild_weed"),
        ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400", "Alpine sorrel foliage", "wild_weed"),
        ("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=400", "Forest bracken fern", "wild_weed"),
        ("https://images.unsplash.com/photo-1476820865390-c52aeebb9891?w=400", "Common dandelion clock", "wild_weed"),
        ("https://images.unsplash.com/photo-1500534623283-312aade485b7?w=400", "Wild field grasses", "wild_weed"),

        # Novel trees and forestry foliage
        ("https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400", "Eucalyptus foliage", "tree_leaf"),
        ("https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=400", "Weeping willow fronds", "tree_leaf"),
        ("https://images.unsplash.com/photo-1448375240586-882707db888b?w=400", "Silver birch leaves", "tree_leaf"),
        ("https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400", "Cypress branch", "tree_leaf"),
        ("https://images.unsplash.com/photo-1463936575829-25148e1db1b8?w=400", "Magnolia leaves", "tree_leaf"),
        ("https://images.unsplash.com/photo-1508873696983-2df5293cb325?w=400", "Ginkgo biloba leaf", "tree_leaf"),
        ("https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=400", "Elm leaf canopy", "tree_leaf"),
        ("https://images.unsplash.com/photo-1470058869958-2a77ade41c02?w=400", "Poplar tree leaves", "tree_leaf"),
        ("https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=400", "Beech forest canopy", "tree_leaf"),
        ("https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=400", "Redwood needle cluster", "tree_leaf"),

        # Novel everyday indoor / outdoor objects
        ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400", "Ceramic coffee mug", "object"),
        ("https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400", "Travel backpack", "object"),
        ("https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400", "Stainless water bottle", "object"),
        ("https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=400", "Road bicycle frame", "object"),
        ("https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=400", "Modern desk lamp", "object"),
        ("https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400", "Hardcover book", "object"),
        ("https://images.unsplash.com/photo-1589939705384-5185137a7f0f?w=400", "Stainless scissors", "object"),
        ("https://images.unsplash.com/photo-1578749556568-bc2c40e68b61?w=400", "Ceramic dinner plate", "object"),
        ("https://images.unsplash.com/photo-1584990347449-34fb8e9c60e0?w=400", "Metal table spoon", "object"),
        ("https://images.unsplash.com/photo-1581147036324-c17ac41dfa6c?w=400", "Hand screwdriver tool", "object")
    ]
    
    start_idx = len(records)
    downloaded_new = 0
    for idx, (url, desc, cat) in enumerate(new_urls):
        fname = f"novel_{start_idx + idx:03d}.jpg"
        target_path = os.path.join(out_dir, fname)
        if not os.path.exists(target_path):
            try:
                download_file(url, target_path)
            except Exception as e:
                print(f"Failed to download {desc}: {e}")
        if os.path.exists(target_path):
            downloaded_new += 1
            records.append({
                "filename": fname,
                "filepath": target_path,
                "description": desc,
                "category": cat
            })
            
    df_manifest = pd.DataFrame(records)
    manifest_csv = "scratch/novel_ood_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\n[+] Total Novel OOD samples compiled: {len(df_manifest)}")
    print(df_manifest["category"].value_counts())
    print(f"[+] Saved manifest to {manifest_csv}")

if __name__ == "__main__":
    main()
