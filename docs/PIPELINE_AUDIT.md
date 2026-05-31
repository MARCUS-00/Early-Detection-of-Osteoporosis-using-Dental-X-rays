# Pipeline Audit — Verified / Partially Verified / Not Verified

This document maps every claimed component of the system to its verification
status based on the available source files:
- `Complete_implementation.ipynb`
- `train_mobilenet.py`
- `test_mobilenet.py`
- `ROI_Extraction.ipynb`
- `Unet_Extraction.ipynb`

---

## VERIFIED (code present and complete in sources)

| Component | Source | Module |
|---|---|---|
| `remove_border(img, threshold=5)` | Complete_implementation.ipynb | `osteo.preprocessing.border` |
| `split_into_patches(img, patch_size=100)` | Complete_implementation.ipynb | `osteo.preprocessing.patches` |
| MobileNetV2 model architecture | train_mobilenet.py | `osteo.classification.model` |
| MobileNetV2 training loop (epochs=20, Adam lr=1e-4, ReduceLROnPlateau only) | train_mobilenet.py | `osteo.classification.train` |
| MobileNetV2 evaluation (per-patch, classification_report) | test_mobilenet.py | `osteo.classification.evaluate` |
| `predict_image()` patch-majority-vote inference | Complete_implementation.ipynb | `osteo.inference.predict_patches` |
| YOLOv8 training (Config A: epochs=50/imgsz=1024/batch=2; Config B: epochs=30/imgsz=640/batch=8) | ROI_Extraction.ipynb | `osteo.roi.yolo_train` |
| `roi_extraction()` left/right box selection | ROI_Extraction.ipynb | `osteo.roi.yolo_extract` |
| `crop_from_labels()` YOLO-label ROI cropping | Unet_Extraction.ipynb | `osteo.roi.crop_from_labels` |
| `directional_gradient()` multi-directional Sobel | Unet_Extraction.ipynb | `osteo.segmentation.masks_classical` |
| `lower_boundary_mask()` gradient threshold + MORPH_OPEN | Unet_Extraction.ipynb | `osteo.segmentation.masks_classical` |
| `upper_boundary_mask()` per-column brightness above lower boundary | Unet_Extraction.ipynb | `osteo.segmentation.masks_classical` |
| `create_final_mask()` lower+upper union | Unet_Extraction.ipynb | `osteo.segmentation.masks_classical` |
| Column-fill and bone-region extraction | Unet_Extraction.ipynb | `osteo.segmentation.masks_classical` |
| `pad_to_fixed_size_center()` centred border pad | Unet_Extraction.ipynb | `osteo.segmentation.dataset` |
| `ROIDatasetPad` PyTorch dataset | Unet_Extraction.ipynb | `osteo.segmentation.dataset` |
| `UNet` architecture (features=[64,128,256,512], sigmoid output) | Unet_Extraction.ipynb | `osteo.segmentation.unet` |
| `train_unet()` BCELoss/Adam 20 epochs, 80/20 split | Unet_Extraction.ipynb | `osteo.segmentation.train_unet` |
| Classifier dataset on-disk layout | On-disk layout `100x100/{train,valid,test}/{Normal,Osteopenia,Osteoporosis}` confirmed from dataset screenshots. Class indices (alphabetical): Normal=0, Osteopenia=1, Osteoporosis=2. | `data/README.md`, `configs/classifier.yaml` |
| Dataset composition (audit-measured) | 13 source X-rays: 3 Normal, 6 Osteopenia, 4 Osteoporosis; ~5,775 augmented 100×100 patches per source; 75,075 total. Source key = filename stem minus trailing `_<digits>`. | `scripts/study_dataset.py`, `data/manifest.csv` |
| Brightness confound (measured) | Brightness-only LOSO baseline: 53.8% per-image (7/13 sources), vs 33.3% chance. Normal class unrecognised by brightness alone (0/3). Per-image mean-std standardisation collapses inter-class brightness to zero (verified). | `scripts/confound_diagnostic.py`, `outputs/confound_report.txt` |
| Channel inconsistency (measured) | ~25% of Osteoporosis patches are 1-channel PNG; Normal and Osteopenia are 100% 3-channel (R=G=B grayscale). Preprocessing normalises all inputs to 3-channel grayscale. | `scripts/confound_diagnostic.py`, `osteo.evaluation.preprocessing` |
| Source-disjoint LOSO CV | 13 folds, one source held out per fold; hard assertion that no source_key appears in both train and eval. Headline metric = per-image (majority vote). | `osteo.evaluation.grouped_cv`, `osteo.evaluation.cv_runner` |
| Per-image standardisation preprocessing | Channel collapse (mean over channels → 3× stack) + per-image (subtract mean, divide by std+1e-7). Applied in both train and eval; replaces rescale=1/255. | `osteo.evaluation.preprocessing` |
| Dataset manifest | `data/manifest.csv`: 75,075 rows, 13 sources verified by assertion. Columns: relative_path, class, source_key, split. | `osteo.evaluation.build_manifest` |

---

## PARTIALLY VERIFIED (present in sources but with known issues)

| Component | Issue | Location |
|---|---|---|
| `roi_extraction()` empty-side handling | Original code had no guard against empty left/right detection lists; a `ValueError` guard was added as the only permitted bugfix. | `osteo.roi.yolo_extract` |
| Model filename consistency | `train_mobilenet.py` saves `"osteoporosis_mobilenetv2.h5"` but `test_mobilenet.py` loaded `"osteoporosis_mobilenet_model.h5"`. Standardised to the trained name. | `osteo.classification.evaluate`, `configs/classifier.yaml` |
| `ROIDataset` (no Pad) | Referenced in Unet_Extraction.ipynb but never defined. Only `ROIDatasetPad` is implemented; all uses of bare `ROIDataset` are a source bug. | `osteo.segmentation.dataset` |
| UNet definition duplication | Defined ~4 times in Unet_Extraction.ipynb. Deduplicated to a single canonical copy; logic unchanged. | `osteo.segmentation.unet` |
| U-Net left-side training divergence | One left-side run in the notebook achieved loss ≈ 2.0254 (diverged). Recorded; not fixed. | `osteo.segmentation.train_unet` |
| Pseudo-ground-truth masks | Masks are classical-CV outputs, NOT human annotations. U-Net was trained on these, not expert labels. | `osteo.segmentation.masks_classical` |

---

## NOT VERIFIED (absent from all source files — placeholders only)

All items below are represented only by placeholder stubs with
`# NOT VERIFIED FROM AVAILABLE FILES` banners and `raise NotImplementedError`.

| Component | Notes | Placeholder location |
|---|---|---|
| End-to-end YOLO→U-Net→classifier pipeline | No wiring code exists in any source file | `osteo.inference.pipeline_full` |
| DenseNet-201 training and comparison | Mentioned in documentation; no code | (no file — would be unverified) |
| ResNeXt-50 training and comparison | Mentioned in documentation; no code | (no file — would be unverified) |
| NASNetMobile training and comparison | Mentioned in documentation; no code | (no file — would be unverified) |
| ViT (vit_b_16) segmentation experiment | Present in notebook but broken/abandoned | `osteo.segmentation.experimental_vit` |
| Dataset construction / labelling code | Code that assigns class labels (Normal/Osteopenia/Osteoporosis) to source images is absent from all source files. Layout is verified; the construction pipeline is not. | `osteo.data.build_dataset` (placeholder) |
| 70/15/15 split ratio and split method | Claimed in documentation; split logic absent from sources. Folder names (train/valid/test) are verified; the code that created them is not. | `data/README.md` |
| Grad-CAM visualisation | Mentioned in docs; no implementation | — |
| Confidence-scored diagnostic reports | Mentioned in docs; no implementation | — |
| Web UI / FastAPI / Flask | Mentioned in docs; no implementation | — |
| Docker containerisation | Mentioned in docs; no implementation | — |
| AWS S3 / cloud integration | Mentioned in docs; no implementation | — |
| EarlyStopping callback | Documented but absent from train_mobilenet.py | — |
| ModelCheckpoint callback | Documented but absent from train_mobilenet.py | — |
| class_weight balancing | Documented but absent from train_mobilenet.py | — |
| Per-patient accuracy | Only per-patch accuracy is computable from available code | — |

---

## Data leakage — CONFIRMED AND RESOLVED

**Audit result (scripts/check_leakage.py, scripts/confound_diagnostic.py):**
All 13 source X-rays have patches split patch-by-patch across train/valid/test.
Every source appears in all three folders with a fixed 80/10/10 ratio
(4,621/577/577 patches). There is **total leakage** — training and test sets
share source patients.

**Resolution applied:**
- The original train/valid/test folder structure is **not used for evaluation**.
- All evaluation uses source-disjoint LOSO cross-validation via `grouped_cv.py`.
- No source_key appears in both train and eval in any fold (hard-asserted).

---

## Accuracy claims — PRIOR FIGURES INVALID

**The 97.28% accuracy and associated F1/precision/recall figures cited in
project documentation are INVALID and must not be reproduced or cited.**

Reasons (measured, not inferred):
1. The split that produced those figures is confirmed patch-level (fully leaked).
2. A brightness-only trivial model achieves 53.8% per-image LOSO, confirming
   the classes are partially separable on a single scalar — brightness confound.
3. ~25% of Osteoporosis patches have a different storage format (1-channel vs
   3-channel), providing a second trivial class shortcut.

**Real evaluation numbers:** see `outputs/cv_report.txt`, produced by running
`scripts/run_cv.py` after GPU training on Kaggle. Numbers are inserted there
only once measured; they are not pre-filled in this document.

**Limitations (from cv_report.txt template):**
- N=13 source X-rays: per-image accuracy is over 13 data points; high variance.
- Brightness confound partially present (53.8% brightness-only baseline).
- 13-patient proof-of-concept; not a clinical validation study.
