# Dataset Card: 10-Crop Leaf Pathology Benchmark

> **Mandatory Disclaimer**: The output of models trained on this system is an **AI-assisted indication, NOT a guaranteed agricultural diagnosis**. Critical agricultural interventions, pesticide deployments, and crop culling must be validated by trained agronomists and laboratory assays.

---

## 1. Overview & Provenance
This dataset card documents the three agricultural image datasets supporting the multi-class leaf disease diagnosis system.

```mermaid
flowchart TD
    subgraph Primary Training & Evaluation
        PV["PlantVillage (Original non-augmented)<br/>9 Crops: Tomato, Potato, Pepper, Maize, Apple, Grape, Peach, Strawberry, Cherry<br/>Lab solid-background | CC BY-SA 4.0"]
        Rice["Philippines Rice Leaf Diseases<br/>1 Crop: Rice (Paddy)<br/>Real-field photography | GNU AGPL 3.0"]
    end
    subgraph Secondary Domain-Shift Benchmark
        PD["PlantDoc (Kayal et al., 2020)<br/>25 Mapped Field Classes (3,996 crops)<br/>STRICTLY EXCLUDED from training | CC BY 4.0"]
    end
    PV --> TrainValTest["Unified Train / Val / Test<br/>(18,789 Images / 37 Classes)"]
    Rice --> TrainValTest
    PD --> ExternalBench["External Domain-Shift Benchmark<br/>(Field Evaluation Only)"]
```

---

## 2. Dataset Citations, Licenses & Acquisition Metadata

### 1. PlantVillage (Primary Lab Dataset)
- **Source**: spMohanty / EPFL & Penn State University.
- **Version**: Original, non-augmented laboratory photography (verified by unique UUID filename hashes).
- **License**: Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **Download Date**: October 2026.
- **Citation**:
  ```bibtex
  @article{mohanty2016using,
    title={Using deep learning for image-based plant disease detection},
    author={Mohanty, Sharada P and Hughes, David P and Salath{\'e}, Marcel},
    journal={Frontiers in plant science},
    volume={7},
    pages={1419},
    year={2016},
    publisher={Frontiers}
  }
  ```
- **Visual Style**: Controlled laboratory conditions. Detached leaves flat on solid gray, white, or black backdrops with uniform illumination.

### 2. Philippines Rice Leaf Disease Dataset (Rice Field Benchmark)
- **Source**: Kaggle repository `shrupyag001/philippines-rice-diseases`.
- **License**: GNU Affero General Public License v3.0 ([AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)).
- **Download Date**: October 2026.
- **Citation**:
  ```bibtex
  @misc{philippines_rice_2023,
    title={Philippines Rice Leaf Diseases Dataset},
    author={Shrupyag et al.},
    year={2023},
    publisher={Kaggle},
    howpublished={\url{https://www.kaggle.com/datasets/shrupyag001/philippines-rice-diseases}}
  }
  ```
- **Visual Style**: Outdoor paddy field photography under natural daylight, direct sun angles, ambient vegetation shadows, and soil backgrounds.

### 3. PlantDoc (Secondary External Domain-Shift Benchmark)
- **Source**: Indian Institute of Technology (IIT) Kharagpur & Roboflow.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Download Date**: October 2026.
- **Citation**:
  ```bibtex
  @inproceedings{singh2020plantdoc,
    title={PlantDoc: a dataset for collaborative computer vision based plant disease detection},
    author={Singh, Davinder and Jain, Naman and Jain, Pranjali and Kayal, Pratik and Kumawat, Sudhakar and Batra, Nipun},
    booktitle={Proceedings of the 7th ACM IKDD CoDS and 25th COMAD},
    pages={249--253},
    year={2020}
  }
  ```
- **Role in Project**: **STRICTLY HELD OUT as an external domain-shift test set**. 3,996 leaf crops across 25 mapped classes. Unmapped categories (e.g. Blueberry, Raspberry, Soybean, Squash) were strictly excluded. Never mixed into training data.

---

## 3. Dataset Inspection & Quality Metrics

- **Total Ingested Images**: 18,789 images
- **Number of Supported Classes**: 37 classes across 10 crops
- **Color Mode**: 100% RGB (3-channel)
- **File Formats**: 100% standard JPEG (`.jpg`/`.jpeg`)
- **Resolution Distribution**:
  - Modal resolution: $256 \times 256$ pixels (PlantVillage standard)
  - Minimum resolution: $128 \times 128$ pixels
  - Maximum resolution: $4000 \times 3000$ pixels (Field photography)
  - All images verified to have short side $\ge 64$ pixels.
- **Near-Duplicate Clustering**:
  - Evaluated with pHash and dHash. Near-duplicate shots of identical leaves were identified and grouped to prevent cross-partition leakage.
  - Zero exact MD5 duplicates present.
  - Full audit logged in [`reports/cleaning_log.csv`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task-222/reports/cleaning_log.csv).
