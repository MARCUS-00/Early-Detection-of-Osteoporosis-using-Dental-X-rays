# Model Weights

Drop `osteoporosis_efficientnetb0.keras` into this folder before running the app.

## File

| File | SHA-256 |
|---|---|
| `osteoporosis_efficientnetb0.keras` | `0ff874482a860b6466882ac25c70655d3f38c0a02d10d1bee98ed24884e172c5` |

## Results (Source-Disjoint LOSO CV, N=13)

| Metric | Value |
|---|---|
| Per-image accuracy | 61.5% (8/13) |
| Macro F1 | 0.594 |
| Normal F1 | 0.500 |
| Osteopenia F1 | 0.615 |
| Osteoporosis F1 | 0.667 |
| Bootstrap 95% CI | [30.8%, 84.6%] |

Baseline (brightness-only): 53.8% accuracy / 0.533 macro-F1.

## Disclaimer

**RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.**
This model was trained on 13 source panoramic X-rays (75,075 augmented 100x100 patches).
Per-image accuracy is measured over 13 data points; confidence intervals are wide.
Do not use for diagnosis.
