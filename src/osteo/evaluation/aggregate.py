"""
Phase E — Honest aggregation and cv_report.txt writer.

Reads per-source majority-vote predictions from all folds and produces:
  outputs/cv_report.txt  — human-readable
  outputs/cv_report.json — machine-readable

Headline metric: PER-IMAGE accuracy (one prediction per source X-ray via majority vote).
Secondary metric: mean ± std per-fold patch accuracy (clearly labelled).

Comparison block:
  - vs 33.3% chance baseline
  - vs 53.8% brightness-only LOSO baseline
  - Whether the 3 Normal sources are correctly classified (brightness-blind test)

No fabricated numbers. All values computed from actual fold predictions.
"""

from __future__ import annotations
import json
from pathlib import Path

CLASSES         = ["Normal", "Osteopenia", "Osteoporosis"]
CHANCE_BASELINE = 1.0 / 3.0          # 33.33%
BRIGHTNESS_LOSO = 7 / 13             # 53.85% — measured in confound_diagnostic.py

NORMAL_SOURCES = {
    "roiant22_1n", "roiant26_1n", "roiant28_1n",
}


def build_cv_report(
    source_preds: list[dict],
    patch_fold_accs: list[float],
    mode: str,
    output_dir: str = "outputs",
) -> Path:
    """
    Build cv_report.txt and cv_report.json.

    Args:
        source_preds:    list of dicts from cv_runner, each with keys:
                         fold, source_key, true_class, pred_class, correct, votes.
        patch_fold_accs: list of per-fold patch-level accuracies (float).
        mode:            "LOSO" or "GROUPKFOLD".
        output_dir:      where to write reports.

    Returns:
        Path to cv_report.txt.
    """
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    report: dict = {}

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("CV REPORT — HONEST EVALUATION")
    log(f"CV mode: {mode}")
    log(f"Total folds: {len(patch_fold_accs)}")
    log("=" * 72)

    # ── Per-source table ──────────────────────────────────────────────────────
    log("\n--- Per-source results (majority vote) ---")
    log(f"{'Fold':>5}  {'Source key':<30}  {'True':<14}  {'Predicted':<14}  {'Correct?':>8}")
    log("-" * 80)

    n_correct = 0
    normal_results: dict[str, bool] = {}

    for row in sorted(source_preds, key=lambda r: (r["fold"], r["source_key"])):
        ok = row["correct"]
        n_correct += int(ok)
        mark = "YES" if ok else "NO "
        log(f"{row['fold']:>5}  {row['source_key']:<30}  {row['true_class']:<14}  "
            f"{row['pred_class']:<14}  {mark:>8}")
        if row["source_key"] in NORMAL_SOURCES:
            normal_results[row["source_key"]] = ok

    n_total = len(source_preds)
    per_image_acc = n_correct / n_total if n_total else float("nan")

    log("-" * 80)
    log(f"Per-image accuracy (majority vote): {n_correct}/{n_total} = "
        f"{per_image_acc:.4f}  ({per_image_acc*100:.1f}%)")

    # ── Per-fold patch accuracy (secondary) ──────────────────────────────────
    if patch_fold_accs:
        arr = np.array(patch_fold_accs)
        log(f"\n--- Per-fold PATCH accuracy (secondary metric, clearly labelled) ---")
        log(f"  Patch accuracy is per individual 100x100 patch; NOT per patient/image.")
        for i, acc in enumerate(patch_fold_accs):
            log(f"  Fold {i:>2}: patch accuracy = {acc:.4f}  ({acc*100:.1f}%)")
        log(f"  Mean patch accuracy across folds: {arr.mean():.4f} +/- {arr.std():.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    n_cls = len(CLASSES)
    cm = [[0]*n_cls for _ in range(n_cls)]
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    for row in source_preds:
        t = cls_idx.get(row["true_class"], -1)
        p = cls_idx.get(row["pred_class"], -1)
        if t >= 0 and p >= 0:
            cm[t][p] += 1

    log(f"\n--- Confusion matrix (rows=true, cols=predicted; per-image) ---")
    col_w = 14
    log(f"  {'True \\ Pred':<18}" + "".join(f"{c:>{col_w}}" for c in CLASSES))
    log("  " + "-" * (18 + col_w * n_cls))
    for i, cls in enumerate(CLASSES):
        log(f"  {cls:<18}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(n_cls)))

    # ── Comparison block ──────────────────────────────────────────────────────
    log("\n--- Comparison against baselines ---")
    log(f"  Chance baseline (uniform random, 3 classes): "
        f"{CHANCE_BASELINE:.4f} ({CHANCE_BASELINE*100:.1f}%)")
    log(f"  Brightness-only LOSO baseline (measured):   "
        f"{BRIGHTNESS_LOSO:.4f} ({BRIGHTNESS_LOSO*100:.1f}%)")
    log(f"  This model (per-image LOSO):                 "
        f"{per_image_acc:.4f} ({per_image_acc*100:.1f}%)")

    if not (per_image_acc != per_image_acc):  # not NaN
        vs_chance = per_image_acc - CHANCE_BASELINE
        vs_bright = per_image_acc - BRIGHTNESS_LOSO
        log(f"\n  Delta vs chance:     {vs_chance:+.4f} ({vs_chance*100:+.1f} pp)")
        log(f"  Delta vs brightness: {vs_bright:+.4f} ({vs_bright*100:+.1f} pp)")

    # ── Normal sources (brightness-blind test) ───────────────────────────────
    log("\n--- Brightness-blind test: Normal sources (key diagnostic check) ---")
    log("  The brightness-only LOSO baseline correctly identifies 0/3 Normal sources.")
    log("  If this model also fails on Normal, it is likely exploiting brightness.")
    log("")
    n_normal_correct = sum(normal_results.values())
    for sk in sorted(NORMAL_SOURCES):
        if sk in normal_results:
            status = "CORRECT" if normal_results[sk] else "WRONG  "
            log(f"  {sk:<30} {status}")
        else:
            log(f"  {sk:<30} (not evaluated in this run)")
    log(f"\n  Normal sources correct: {n_normal_correct}/{len(NORMAL_SOURCES)}")
    log(f"  (Brightness-only baseline: 0/{len(NORMAL_SOURCES)})")

    # ── Limitations ──────────────────────────────────────────────────────────
    log("\n--- Limitations ---")
    log(f"  1. N=13 source X-rays; per-image accuracy is over 13 data points.")
    log(f"     Results have high variance and should not be over-interpreted.")
    log(f"  2. Brightness confound partially present (53.8% brightness-only baseline).")
    log(f"     Per-image standardization applied; residual confound not fully quantified.")
    log(f"  3. All ~75,075 patches derive from 13 patients; this is a proof-of-concept,")
    log(f"     not a clinical validation study.")
    log(f"  4. The prior patch-level 97.28%/F1 figures (leaked split) are INVALID.")
    log(f"     They are NOT reproduced anywhere in this evaluation.")
    log(f"  5. pipeline_full.py remains a NotImplementedError placeholder.")

    log("\n" + "=" * 72)

    # ── Write outputs ─────────────────────────────────────────────────────────
    txt_path  = out / "cv_report.txt"
    json_path = out / "cv_report.json"

    report = {
        "cv_mode": mode,
        "n_folds": len(patch_fold_accs),
        "per_image_accuracy": per_image_acc,
        "n_correct_sources": n_correct,
        "n_total_sources": n_total,
        "confusion_matrix": cm,
        "classes": CLASSES,
        "patch_fold_accuracies": patch_fold_accs,
        "patch_mean_acc": float(np.mean(patch_fold_accs)) if patch_fold_accs else None,
        "patch_std_acc": float(np.std(patch_fold_accs)) if patch_fold_accs else None,
        "normal_sources_correct": n_normal_correct,
        "normal_sources_total": len(NORMAL_SOURCES),
        "chance_baseline": CHANCE_BASELINE,
        "brightness_loso_baseline": BRIGHTNESS_LOSO,
        "source_predictions": source_preds,
        "invalid_prior_result": "97.28%/F1 from leaked patch-level split — DO NOT USE",
    }

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReports written:")
    print(f"  {txt_path.resolve()}")
    print(f"  {json_path.resolve()}")
    return txt_path
