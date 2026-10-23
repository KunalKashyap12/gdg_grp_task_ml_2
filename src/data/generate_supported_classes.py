"""
Generates data/supported_classes.json from config.yaml.
Enforces the naming convention Crop___Disease and provides
class_id, crop, disease, display_name, and short plain-language description.
"""

import os
import json
import yaml


def generate_supported_classes(config_path="config.yaml", output_path="data/supported_classes.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    crops_cfg = cfg.get("crops", {})
    classes_list = []
    class_id = 0

    for crop_name, crop_info in crops_cfg.items():
        crop_display = crop_info.get("display_name", crop_name)
        diseases = crop_info.get("diseases", [])
        for d in diseases:
            disease_name = d["name"]
            disease_display = d.get("display_name", disease_name.replace("_", " "))
            desc = d.get("description", "")
            standard_label = f"{crop_name}___{disease_name}"
            
            entry = {
                "class_id": class_id,
                "label": standard_label,
                "crop": crop_name,
                "disease": disease_name,
                "display_name": f"{crop_display}: {disease_display}",
                "description": desc,
                "is_healthy": (disease_name.lower() == "healthy")
            }
            classes_list.append(entry)
            class_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)

    print(f"[+] Successfully generated {len(classes_list)} supported classes in {output_path}")
    return classes_list


if __name__ == "__main__":
    generate_supported_classes()
