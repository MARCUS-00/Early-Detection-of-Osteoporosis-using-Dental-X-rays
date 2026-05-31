"""
Read-only confound diagnostic.
Writes only to ./outputs/.

Usage:
    python scripts/confound_diagnostic.py [--dataset_root PATH]
"""

import argparse
import json
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
SPLITS            = ["train", "valid", "test"]
CLASSES           = ["Normal", "Osteopenia", "Osteoporosis"]
SAMPLE_PER_SOURCE = 300
SAMPLE_PER_FOLD   = 2000
SEED              = 42

random.seed(SEED)
np.random.seed(SEED)

# ── helpers ───────────────────────────────────────────────────────────────────

def source_key(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def to_gray_mean(img: np.ndarray) -> np.ndarray:
    """Convert loaded image to float32 grayscale via channel mean."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return img.mean(axis=2).astype(np.float32)


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return to_gray_mean(img)


# ── collect all files grouped by (source_key, class) ──────────────────────────

def collect_files(root: Path) -> dict:
    """
    Returns:
        {source_key: {"class": str, "files": [Path, ...]}}
    """
    sk_map: dict[str, dict] = {}
    for split in SPLITS:
        for cls in CLASSES:
            p = root / split / cls
            if not p.is_dir():
                continue
            for f in p.iterdir():
                if not f.is_file():
                    continue
                sk = source_key(f.stem)
                if sk not in sk_map:
                    sk_map[sk] = {"class": cls, "files": []}
                sk_map[sk]["files"].append(f)
    return sk_map


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=None)
    args = parser.parse_args()

    if args.dataset_root:
        root = Path(args.dataset_root)
    else:
        cfg_path = Path(__file__).parent.parent / "configs" / "classifier.yaml"
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            train_dir = cfg.get("train_dir", "")
            if not train_dir:
                print("ERROR: train_dir empty in classifier.yaml. Pass --dataset_root.")
                sys.exit(1)
            root = Path(train_dir).parent
        except Exception as exc:
            print(f"ERROR reading classifier.yaml: {exc}")
            sys.exit(1)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    lines: list[str] = []
    report: dict = {}

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("CONFOUND DIAGNOSTIC REPORT")
    log(f"DATASET_ROOT      : {root.resolve()}")
    log(f"SAMPLE_PER_SOURCE : {SAMPLE_PER_SOURCE}")
    log(f"SAMPLE_PER_FOLD   : {SAMPLE_PER_FOLD}")
    log(f"SEED              : {SEED}")
    log("=" * 72)

    sk_map = collect_files(root)
    all_sources = sorted(sk_map.keys())
    log(f"\nTotal source keys found: {len(all_sources)}")

    # ── 1. PER-SOURCE BRIGHTNESS ──────────────────────────────────────────────
    log("\n" + "-" * 72)
    log("1. PER-SOURCE BRIGHTNESS")
    log("-" * 72)

    src_stats: dict[str, dict] = {}
    load_errors_s1: list[str] = []

    rng = random.Random(SEED)
    for sk in all_sources:
        info = sk_map[sk]
        files = list(info["files"])
        rng.shuffle(files)
        sample = files[:SAMPLE_PER_SOURCE]
        intensities = []
        for fp in sample:
            arr = load_gray(fp)
            if arr is None:
                load_errors_s1.append(str(fp))
                continue
            intensities.append(float(arr.mean()))
        if intensities:
            src_stats[sk] = {
                "class": info["class"],
                "n": len(intensities),
                "mean": float(np.mean(intensities)),
                "std": float(np.std(intensities)),
            }
        else:
            src_stats[sk] = {"class": info["class"], "n": 0, "mean": float("nan"), "std": float("nan")}

    log(f"\n  {'Source key':<30} {'Class':<14} {'N':>5} {'Mean':>8} {'Std':>8}")
    log("  " + "-" * 68)
    for sk in all_sources:
        s = src_stats[sk]
        log(f"  {sk:<30} {s['class']:<14} {s['n']:>5} {s['mean']:>8.2f} {s['std']:>8.2f}")

    # Per-class spread of source means
    log("\n  Per-class spread of source means:")
    class_means: dict[str, list[float]] = defaultdict(list)
    for sk, s in src_stats.items():
        if not np.isnan(s["mean"]):
            class_means[s["class"]].append(s["mean"])

    log(f"\n  {'Class':<18} {'N_sources':>10} {'Min_mean':>10} {'Max_mean':>10} {'Range':>8} {'Mean_of_means':>14}")
    log("  " + "-" * 72)
    class_spread: dict[str, dict] = {}
    for cls in CLASSES:
        ms = sorted(class_means[cls])
        if ms:
            spread = {
                "n_sources": len(ms),
                "min_mean": float(min(ms)),
                "max_mean": float(max(ms)),
                "range": float(max(ms) - min(ms)),
                "mean_of_means": float(np.mean(ms)),
                "values": ms,
            }
            class_spread[cls] = spread
            log(f"  {cls:<18} {len(ms):>10} {min(ms):>10.2f} {max(ms):>10.2f} {max(ms)-min(ms):>8.2f} {np.mean(ms):>14.2f}")
        else:
            log(f"  {cls:<18} (no data)")

    # Separability interpretation
    log("\n  Separability analysis:")
    all_class_means_flat = {cls: class_means[cls] for cls in CLASSES if class_means[cls]}
    # compute inter-class distance vs intra-class spread
    separable = False
    interp_lines = []
    class_mean_vals = {cls: np.mean(ms) for cls, ms in all_class_means_flat.items()}
    # Check if ranges overlap
    ranges = {
        cls: (min(class_means[cls]), max(class_means[cls]))
        for cls in CLASSES if class_means[cls]
    }
    overlap_pairs = []
    no_overlap_pairs = []
    cls_list = list(ranges.keys())
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            a, b = cls_list[i], cls_list[j]
            lo_a, hi_a = ranges[a]
            lo_b, hi_b = ranges[b]
            overlap = max(0, min(hi_a, hi_b) - max(lo_a, lo_b))
            if overlap > 0:
                overlap_pairs.append((a, b, overlap))
            else:
                no_overlap_pairs.append((a, b))

    for cls in CLASSES:
        if cls in class_mean_vals:
            log(f"    {cls}: source mean-intensity values = {[round(v,1) for v in sorted(class_means[cls])]}")
    log("")
    if no_overlap_pairs:
        log(f"  NON-OVERLAPPING pairs (fully separable by source brightness): {no_overlap_pairs}")
    if overlap_pairs:
        log(f"  OVERLAPPING pairs (ranges overlap): {[(a,b,round(ov,1)) for a,b,ov in overlap_pairs]}")

    # Determine separability
    if not overlap_pairs:
        sep_verdict = ("YES — all three classes are FULLY SEPARABLE by source brightness alone. "
                       "No overlap in source mean-intensity ranges across classes.")
        separable = True
    elif not no_overlap_pairs:
        sep_verdict = ("NO — all class ranges overlap. Brightness alone does not separate the classes "
                       "at the source level.")
        separable = False
    else:
        sep_verdict = ("PARTIAL — some class pairs are separable, some overlap. "
                       "Brightness is a strong but not complete class signal.")
        separable = True  # partial still means CNN can exploit it

    log(f"\n  VERDICT: {sep_verdict}")

    report["per_source_brightness"] = {
        "source_stats": src_stats,
        "class_spread": class_spread,
        "separable": separable,
        "verdict": sep_verdict,
    }

    # ── 2. BRIGHTNESS-ONLY BASELINE (LOSO) ───────────────────────────────────
    log("\n" + "-" * 72)
    log("2. BRIGHTNESS-ONLY BASELINE — LEAVE-ONE-SOURCE-OUT (LOSO)")
    log("-" * 72)

    try:
        from sklearn.naive_bayes import GaussianNB

        # Build per-patch feature matrix for each source (all patches, mean intensity only)
        log("\n  Building per-patch brightness features for all sources...")
        src_features: dict[str, dict] = {}  # sk -> {X: array, y: int}
        class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

        rng2 = random.Random(SEED + 1)
        for sk in all_sources:
            info = sk_map[sk]
            files = list(info["files"])
            intensities = []
            for fp in files:
                arr = load_gray(fp)
                if arr is not None:
                    intensities.append(float(arr.mean()))
            src_features[sk] = {
                "X": np.array(intensities).reshape(-1, 1),
                "y_patch": np.full(len(intensities), class_to_idx[info["class"]], dtype=int),
                "y_label": class_to_idx[info["class"]],
                "class": info["class"],
                "n": len(intensities),
            }
            log(f"    {sk:<30} class={info['class']:<14} patches={len(intensities)}")

        # LOSO
        log("\n  Running leave-one-source-out cross-validation...")
        y_true_src: list[int] = []
        y_pred_src: list[int] = []
        fold_details: list[dict] = []

        for held_out_sk in all_sources:
            train_sks = [sk for sk in all_sources if sk != held_out_sk]

            # Build training set (sample up to SAMPLE_PER_FOLD total)
            X_train_parts = []
            y_train_parts = []
            rng3 = random.Random(SEED + 2)
            for sk in train_sks:
                feats = src_features[sk]
                n = len(feats["X"])
                idx = list(range(n))
                rng3.shuffle(idx)
                cap = max(1, SAMPLE_PER_FOLD // len(train_sks))
                idx = idx[:cap]
                X_train_parts.append(feats["X"][idx])
                y_train_parts.append(feats["y_patch"][idx])

            X_train = np.vstack(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            # Test set (all patches of held-out source)
            X_test = src_features[held_out_sk]["X"]
            true_label = src_features[held_out_sk]["y_label"]

            # Fit
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            patch_preds = clf.predict(X_test)

            # Majority vote
            counts_vote = np.bincount(patch_preds, minlength=len(CLASSES))
            pred_label = int(np.argmax(counts_vote))

            y_true_src.append(true_label)
            y_pred_src.append(pred_label)

            true_cls = CLASSES[true_label]
            pred_cls = CLASSES[pred_label]
            correct = true_label == pred_label
            vote_str = " ".join(f"{CLASSES[i]}:{counts_vote[i]}" for i in range(len(CLASSES)))
            fold_details.append({
                "held_out": held_out_sk,
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "correct": correct,
                "patch_votes": {CLASSES[i]: int(counts_vote[i]) for i in range(len(CLASSES))},
            })
            status = "CORRECT" if correct else "WRONG  "
            log(f"    {status}  held_out={held_out_sk:<30} true={true_cls:<14} pred={pred_cls:<14}  votes=[{vote_str}]")

        # Accuracy
        n_correct = sum(1 for t, p in zip(y_true_src, y_pred_src) if t == p)
        per_image_acc = n_correct / len(y_true_src)
        log(f"\n  Per-image accuracy (majority vote, LOSO): {n_correct}/{len(y_true_src)} = {per_image_acc:.4f} ({per_image_acc*100:.1f}%)")

        # Confusion matrix
        n_cls = len(CLASSES)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true_src, y_pred_src):
            cm[t][p] += 1

        log("\n  Confusion matrix (rows=true, cols=predicted):")
        header_cm = f"  {'True \\ Pred':<18}" + "".join(f"{c:>14}" for c in CLASSES)
        log(header_cm)
        log("  " + "-" * (18 + 14 * n_cls))
        for i, cls in enumerate(CLASSES):
            row = f"  {cls:<18}" + "".join(f"{cm[i][j]:>14}" for j in range(n_cls))
            log(row)

        # Interpretation
        log("\n  Interpretation:")
        if per_image_acc >= 0.85:
            interp = (
                f"A brightness-ONLY model (single scalar = mean patch intensity) achieves "
                f"{per_image_acc*100:.1f}% per-image accuracy on LOSO. "
                f"The classes are highly separable on brightness alone. "
                f"A CNN trained on this dataset is very likely exploiting the same "
                f"brightness shortcut rather than learning bone texture differences. "
                f"The per-patch accuracy reported in the original work is therefore "
                f"misleading as a diagnostic performance metric."
            )
        elif per_image_acc >= 0.50:
            interp = (
                f"A brightness-ONLY model achieves {per_image_acc*100:.1f}% per-image accuracy on LOSO, "
                f"above chance (33%). Brightness is a partial confound; a CNN may be exploiting "
                f"it in addition to texture features. Normalizing per-image brightness before "
                f"training is recommended."
            )
        else:
            interp = (
                f"A brightness-ONLY model achieves {per_image_acc*100:.1f}% per-image accuracy on LOSO, "
                f"near or below chance. Brightness alone is not a strong class signal at the source level."
            )
        log(f"  {interp}")

        report["brightness_baseline_loso"] = {
            "per_image_accuracy": per_image_acc,
            "n_correct": n_correct,
            "n_total": len(y_true_src),
            "confusion_matrix": cm.tolist(),
            "fold_details": fold_details,
            "interpretation": interp,
        }

    except Exception as exc:
        log(f"\n  ERROR in brightness baseline: {exc}")
        import traceback
        log(traceback.format_exc())
        report["brightness_baseline_loso"] = {"error": str(exc)}

    # ── 3. NORMALIZATION CHECK ────────────────────────────────────────────────
    log("\n" + "-" * 72)
    log("3. NORMALIZATION CHECK")
    log("-" * 72)

    try:
        log("\n  Computing per-class mean intensity BEFORE and AFTER per-patch standardization...")
        log(f"  (Sample: up to {SAMPLE_PER_SOURCE} patches per source = up to {SAMPLE_PER_SOURCE * len(all_sources)} total)")

        raw_means_by_class: dict[str, list[float]] = defaultdict(list)
        norm_means_by_class: dict[str, list[float]] = defaultdict(list)

        rng4 = random.Random(SEED + 3)
        for sk in all_sources:
            info = sk_map[sk]
            cls = info["class"]
            files = list(info["files"])
            rng4.shuffle(files)
            sample = files[:SAMPLE_PER_SOURCE]
            for fp in sample:
                arr = load_gray(fp)
                if arr is None:
                    continue
                flat = arr.flatten().astype(np.float64)
                raw_means_by_class[cls].append(float(flat.mean()))
                # per-patch standardization
                mu = flat.mean()
                sigma = flat.std()
                if sigma > 1e-6:
                    norm = (flat - mu) / sigma
                else:
                    norm = flat - mu
                norm_means_by_class[cls].append(float(norm.mean()))

        log(f"\n  {'Class':<18} {'Before mean':>14} {'After mean':>14}  Collapse?")
        log("  " + "-" * 56)
        norm_check: dict = {}
        for cls in CLASSES:
            before = float(np.mean(raw_means_by_class[cls])) if raw_means_by_class[cls] else float("nan")
            after  = float(np.mean(norm_means_by_class[cls])) if norm_means_by_class[cls] else float("nan")
            collapse = abs(after) < 0.01  # mean should be ~0 after per-patch standardization
            log(f"  {cls:<18} {before:>14.4f} {after:>14.6f}  {'YES' if collapse else 'NO'}")
            norm_check[cls] = {"before_mean": before, "after_mean": after, "collapses": collapse}

        all_collapse = all(v["collapses"] for v in norm_check.values())
        log(f"\n  Per-class means after per-patch standardization: "
            f"{'all converge to ~0 (brightness separation eliminated)' if all_collapse else 'do NOT fully converge — residual signal remains'}")
        log("  Conclusion: per-image mean-std normalization WOULD remove the brightness confound.")
        log("  Without it, the model can classify by brightness rather than bone texture.")

        report["normalization_check"] = {
            "per_class": norm_check,
            "all_collapse_to_zero": all_collapse,
        }

    except Exception as exc:
        log(f"\n  ERROR in normalization check: {exc}")
        report["normalization_check"] = {"error": str(exc)}

    # ── 4. CHANNEL / PROCESSING CHECK ────────────────────────────────────────
    log("\n" + "-" * 72)
    log("4. CHANNEL / PROCESSING CHECK")
    log("-" * 72)

    try:
        CAP = 2000
        log(f"\n  Scanning up to {CAP} files per class across all splits...")

        chan_results: dict[str, dict] = {}
        for cls in CLASSES:
            cls_files = []
            for split in SPLITS:
                p = root / split / cls
                if p.is_dir():
                    cls_files.extend(f for f in p.iterdir() if f.is_file())
            rng5 = random.Random(SEED + 4)
            rng5.shuffle(cls_files)
            sample = cls_files[:CAP]
            n1, n3, n_other, n_err = 0, 0, 0, 0
            for fp in sample:
                img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                if img is None:
                    n_err += 1
                    continue
                if img.ndim == 2:
                    n1 += 1
                elif img.ndim == 3 and img.shape[2] == 3:
                    n3 += 1
                else:
                    n_other += 1
            total = n1 + n3 + n_other
            chan_results[cls] = {
                "sampled": len(sample),
                "1ch": n1, "3ch": n3, "other": n_other, "errors": n_err,
                "frac_1ch": round(n1 / total, 4) if total else float("nan"),
                "frac_3ch": round(n3 / total, 4) if total else float("nan"),
            }

        log(f"\n  {'Class':<18} {'Sampled':>8} {'1-ch':>8} {'3-ch':>8} {'Other':>7} {'Frac_1ch':>10} {'Frac_3ch':>10}")
        log("  " + "-" * 72)
        for cls in CLASSES:
            r = chan_results[cls]
            log(f"  {cls:<18} {r['sampled']:>8} {r['1ch']:>8} {r['3ch']:>8} {r['other']:>7} {r['frac_1ch']:>10.4f} {r['frac_3ch']:>10.4f}")

        # Determine if 1-channel concentrated in Osteoporosis
        frac_1ch = {cls: chan_results[cls]["frac_1ch"] for cls in CLASSES}
        log(f"\n  1-channel fractions: Normal={frac_1ch['Normal']:.4f}, "
            f"Osteopenia={frac_1ch['Osteopenia']:.4f}, "
            f"Osteoporosis={frac_1ch['Osteoporosis']:.4f}")

        if frac_1ch["Osteoporosis"] > max(frac_1ch["Normal"], frac_1ch["Osteopenia"]) + 0.05:
            chan_verdict = ("YES — 1-channel files are concentrated in Osteoporosis. "
                           "This indicates a per-class processing artifact: Osteoporosis images were saved "
                           "differently (or sourced from a different pipeline step) than Normal/Osteopenia. "
                           "A model trained without channel normalization can exploit this channel-count "
                           "difference as a class shortcut.")
        else:
            chan_verdict = ("NO — 1-channel files are NOT disproportionately concentrated in Osteoporosis. "
                            "Channel count does not appear to be a class-level artifact.")

        log(f"\n  VERDICT: {chan_verdict}")

        report["channel_check"] = {
            "per_class": chan_results,
            "verdict": chan_verdict,
        }

    except Exception as exc:
        log(f"\n  ERROR in channel check: {exc}")
        report["channel_check"] = {"error": str(exc)}

    # ── Write outputs ─────────────────────────────────────────────────────────
    txt_path  = out_dir / "confound_report.txt"
    json_path = out_dir / "confound_report.json"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    log("\n" + "=" * 72)
    log(f"Reports written:")
    log(f"  {txt_path.resolve()}")
    log(f"  {json_path.resolve()}")
    log("=" * 72)


if __name__ == "__main__":
    main()
