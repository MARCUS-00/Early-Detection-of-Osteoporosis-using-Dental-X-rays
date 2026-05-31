# Early Detection of Osteoporosis using Dental X-rays

A research codebase for screening osteoporosis from dental X-rays using
MobileNetV2 patch classification. The repository is organised into four
**independent** stages; they do not form a connected pipeline.

---

## Important disclaimers

- **The end-to-end YOLO→U-Net→classifier pipeline is NOT implemented.**
  The actual deployed inference path is patch-based majority voting with
  no YOLO and no U-Net (`osteo.inference.predict_patches`).
- **The prior 97.28% accuracy / F1 figures are INVALID and must not be used.**
  Measured audit result: the original train/valid/test split is patch-level
  (random), meaning every one of the 13 source X-rays has patches in all three
  folds simultaneously. All prior metrics are inflated by this leakage and
  carry no diagnostic validity.
- **Dataset: 13 source X-rays, ~75,075 augmented 100×100 patches.**
  Sources: 3 Normal, 6 Osteopenia, 4 Osteoporosis (~5,775 patches each).
  Evaluation now uses source-disjoint Leave-One-Source-Out (LOSO) CV;
  headline metric is per-image (per-source) via majority vote.
- **Brightness confound documented.** A brightness-only LOSO baseline scores
  53.8% per-image (7/13) versus 33.3% chance. Per-image mean-std
  standardization is applied to all images to address this; whether the
  CNN learns beyond brightness is the key diagnostic question.
- **Channel inconsistency in Osteoporosis.** ~25% of Osteoporosis patches
  are stored as 1-channel; the preprocessing pipeline normalises all inputs
  to 3-channel grayscale before training.
- **Real CV numbers: see outputs/cv_report.txt** (generated after Kaggle
  training completes; not fabricated here).
- **Pseudo-ground-truth masks.** The U-Net was trained on masks generated
  by classical computer vision (Sobel + morphology), not human annotations.
- This code is a research artefact (13-source proof-of-concept), not a
  clinical tool.

---

## Four independent stages

### Stage 1 — Preprocessing
- `remove_border(img, threshold=5)`: removes black border via binary threshold
  and bounding-rect crop.
- `split_into_patches(img, patch_size=100)`: tiles the image into 100×100
  non-overlapping patches; border patches that are smaller are discarded.

**Source:** `Complete_implementation.ipynb`

### Stage 2 — ROI Detection (YOLOv8)
- Detects cortex ROI bounding boxes (class: `cortex_roi`) using YOLOv8n.
- Two training configurations exist in the source:
  - Config A (default): epochs=50, imgsz=1024, batch=2
  - Config B: epochs=30, imgsz=640, batch=8
- `roi_extraction()` splits detections into left/right sides by image midpoint
  and selects the nearest-to-midline box on each side.
- Dataset: Roboflow YOLOv8 export (not distributed).

**Source:** `ROI_Extraction.ipynb`

### Stage 3 — Segmentation (U-Net)
- Classical-CV pseudo-masks: `directional_gradient`, `lower_boundary_mask`,
  `upper_boundary_mask`, `create_final_mask`. These are NOT human annotations.
- `ROIDatasetPad`: PyTorch dataset with centre-pad to 100×100.
- `UNet(in_channels=3, out_channels=1, features=[64,128,256,512])`: standard
  U-Net trained with BCELoss, Adam (lr=1e-3), 20 epochs.
- Saves: `unet_roi_left.pth`, `unet_roi_right.pth`, `unet_roi_general.pth`.
- Note: one left-side training run diverged (loss ≈ 2.0254); this is not fixed.

**Source:** `Unet_Extraction.ipynb`

### Stage 4 — Classification (MobileNetV2)
- Dataset: 13 source X-rays, ~75,075 patches across Normal/Osteopenia/Osteoporosis.
  On-disk layout: `100x100/{train,valid,test}/{Normal,Osteopenia,Osteoporosis}` (verified).
  **The original train/valid/test split is patch-level (leaked). It is NOT used for
  evaluation.** All evaluation uses source-disjoint LOSO CV (`osteo.evaluation`).
- Preprocessing (NEW — addresses measured confounds):
  - Channel normalisation: all images loaded as RGB; preprocessing_function
    collapses to grayscale mean and stacks 3×, erasing 1ch/3ch inconsistency.
  - Brightness normalisation: per-image mean-std standardisation; replaces
    rescale=1/255. Collapses inter-class brightness gap (verified by audit).
  - Train augmentation: rotation20/shift0.2/shear0.2/zoom0.2/hflip (unchanged).
- Input: 100×100 patches at (128,128) resize.
- MobileNetV2 (ImageNet weights) fine-tuned: last 20 layers unfrozen.
- Architecture: base → GlobalAveragePooling2D → BatchNorm → Dense(128) →
  Dropout(0.5) → Dense(3, softmax). **UNCHANGED from source.**
- Callbacks: **only** ReduceLROnPlateau(monitor="val_loss", factor=0.5,
  patience=3). EarlyStopping, ModelCheckpoint, and class_weight are **not**
  implemented (absent from source).
- Headline metric: per-image (per-source) LOSO accuracy via majority vote.
  Secondary: mean±std per-fold patch accuracy (clearly labelled as secondary).
- Real numbers: see `outputs/cv_report.txt` (produced after training).
- **Prior 97.28% / F1 figures: INVALID (leaked split) — not reproduced.**

**Source:** `train_mobilenet.py`, `test_mobilenet.py` + `osteo.evaluation`

---

## Known issues and discrepancies

| Issue | Detail |
|---|---|
| Filename mismatch | `test_mobilenet.py` loaded `"osteoporosis_mobilenet_model.h5"`; `train_mobilenet.py` saved `"osteoporosis_mobilenetv2.h5"`. Standardised to the trained name throughout. |
| Empty-side guard | `roi_extraction()` in the source notebook had no guard against empty left/right detection lists. A `ValueError` with a descriptive message was added as the only permitted fix. |
| `ROIDataset` undefined | The notebook references `ROIDataset` (without Pad) which is never defined. Only `ROIDatasetPad` is implemented. |
| UNet duplicated ~4× | The notebook contains ~4 copies of the UNet definition. Deduplicated to one canonical copy without logic changes. |
| Pseudo-ground-truth | U-Net masks are classical-CV outputs, not manual annotations. |

---

## NOT VERIFIED items

The following are mentioned in project documentation but have **no
implementation** in any source file. They are represented as placeholder
stubs (`raise NotImplementedError`) in this repository:

- End-to-end YOLO→U-Net→classifier pipeline (`osteo.inference.pipeline_full`)
- DenseNet-201, ResNeXt-50, NASNetMobile training and comparison
- ViT (vit_b_16) segmentation experiment (broken/abandoned in notebook)
- 3-class dataset label-assignment step
- 70/15/15 train/val/test split logic (provenance unknown)
- Grad-CAM visualisation
- Confidence-scored diagnostic reports
- Web UI / FastAPI / Flask
- Docker containerisation
- AWS S3 / cloud integration
- EarlyStopping, ModelCheckpoint, class_weight (documented; not in code)

See `docs/PIPELINE_AUDIT.md` for the full verified/not-verified mapping.

---

## Installation

```bash
# From the repo root (directory name contains spaces; pip install -e works via pyproject.toml)
pip install -e .
pip install -r requirements.txt
```

After installation, `import osteo` should resolve without error.

> **Note:** `tensorflow` is left unpinned in `requirements.txt` because
> the version used in the source environment was not recorded. Install a
> TensorFlow version compatible with Python 3.12.

---

## Quick start

```python
# Patch-based majority-vote prediction (the only verified inference path)
from osteo.inference.predict_patches import predict_image

label = predict_image(
    img_path="path/to/xray.png",
    model_path="osteoporosis_mobilenetv2.h5",
    class_names=["Normal", "Osteopenia", "Osteoporosis"],
)
print(label)
```

```bash
# CLI equivalents
python scripts/train_classifier.py --train-dir path/to/100x100/train \
                                   --val-dir   path/to/100x100/valid
python scripts/evaluate_classifier.py --test-dir path/to/100x100/test
python scripts/run_inference.py path/to/xray.png
```

---

## Repository layout

```
Early Detection of Osteoporosis using Dental X-rays/
├── configs/                  # Hyperparameter YAML files (no invented values)
├── data/README.md            # Verified layout (100x100/{train,valid,test}/…); construction = NOT VERIFIED
├── models/.gitkeep           # Weight files not distributed
├── notebooks/                # Original notebooks (preserved verbatim)
├── src/osteo/                # Python package (import name: osteo)
│   ├── preprocessing/        # border.py, patches.py
│   ├── roi/                  # yolo_train, yolo_extract, crop_from_labels
│   ├── segmentation/         # masks_classical, dataset, unet, train_unet,
│   │                         #   experimental_vit (NOT VERIFIED / ABANDONED)
│   ├── classification/       # model, train, evaluate
│   ├── inference/            # predict_patches (VERIFIED), pipeline_full (NOT VERIFIED)
│   └── utils/
├── scripts/                  # Thin argparse wrappers; no business logic
├── tests/                    # Synthetic-data unit tests (no dataset files needed)
└── docs/PIPELINE_AUDIT.md    # Full verified/not-verified mapping
```
