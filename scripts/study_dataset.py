"""
Read-only dataset study.
Writes only to ./outputs/.

Usage:
    python scripts/study_dataset.py [--dataset_root PATH]

If --dataset_root is omitted, infers from configs/classifier.yaml (train_dir/../).
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# â”€â”€ config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SPLITS          = ["train", "valid", "test"]
CLASSES         = ["Normal", "Osteopenia", "Osteoporosis"]
SAMPLE_PER_CLASS = 800
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import re

def source_key(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def all_image_files(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=None)
    args = parser.parse_args()

    # resolve dataset root
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
    log("DATASET STUDY REPORT")
    log(f"DATASET_ROOT     : {root.resolve()}")
    log(f"SAMPLE_PER_CLASS : {SAMPLE_PER_CLASS}")
    log(f"SEED             : {SEED}")
    log("=" * 72)

    # â”€â”€ 1. STRUCTURE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("1. STRUCTURE")
    log("â”€" * 72)

    struct = {}
    missing_folders = []
    unexpected_items = []
    all_extensions: set[str] = set()

    for split in SPLITS:
        split_path = root / split
        if not split_path.is_dir():
            missing_folders.append(str(split_path))
            log(f"  MISSING split folder : {split_path}")
            continue
        for cls in CLASSES:
            cls_path = split_path / cls
            if not cls_path.is_dir():
                missing_folders.append(str(cls_path))
                log(f"  MISSING class folder : {cls_path}")
            else:
                log(f"  EXISTS  {split}/{cls}")
                struct[f"{split}/{cls}"] = True
        # check for unexpected items
        for item in split_path.iterdir():
            if item.name not in CLASSES:
                unexpected_items.append(str(item))
                log(f"  UNEXPECTED item in {split}: {item.name}")

    # collect extensions
    for split in SPLITS:
        for cls in CLASSES:
            p = root / split / cls
            if p.is_dir():
                for f in p.iterdir():
                    if f.is_file():
                        all_extensions.add(f.suffix.lower())

    log(f"\n  File extensions found: {sorted(all_extensions)}")
    log(f"  Missing folders      : {missing_folders or 'none'}")
    log(f"  Unexpected items     : {unexpected_items or 'none'}")

    report["structure"] = {
        "missing_folders": missing_folders,
        "unexpected_items": unexpected_items,
        "extensions": sorted(all_extensions),
    }

    # â”€â”€ 2. COUNTS (ALL files) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("2. COUNTS (all files)")
    log("â”€" * 72)

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            p = root / split / cls
            if p.is_dir():
                counts[split][cls] = sum(1 for f in p.iterdir() if f.is_file())
            else:
                counts[split][cls] = 0

    # header
    col_w = 14
    header = f"  {'Split':<8}" + "".join(f"{c:>{col_w}}" for c in CLASSES) + f"{'Total':>{col_w}}"
    log(header)
    log("  " + "-" * (8 + col_w * (len(CLASSES) + 1)))
    split_totals = {}
    class_totals = {cls: 0 for cls in CLASSES}
    for split in SPLITS:
        row_total = sum(counts[split].values())
        split_totals[split] = row_total
        row = f"  {split:<8}" + "".join(f"{counts[split][cls]:>{col_w},}" for cls in CLASSES) + f"{row_total:>{col_w},}"
        log(row)
        for cls in CLASSES:
            class_totals[cls] += counts[split][cls]
    grand_total = sum(class_totals.values())
    log("  " + "-" * (8 + col_w * (len(CLASSES) + 1)))
    log(f"  {'Total':<8}" + "".join(f"{class_totals[cls]:>{col_w},}" for cls in CLASSES) + f"{grand_total:>{col_w},}")

    report["counts"] = {
        "per_split_class": counts,
        "class_totals": class_totals,
        "split_totals": split_totals,
        "grand_total": grand_total,
    }

    # â”€â”€ 3. SOURCE GROUPING (ALL files) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("3. SOURCE GROUPING (all files)")
    log("â”€" * 72)

    # {source_key: {split: count, "class": cls}}
    # We'll build: source_key -> {split -> int, class -> str}
    sk_info: dict[str, dict] = {}

    for split in SPLITS:
        for cls in CLASSES:
            p = root / split / cls
            if not p.is_dir():
                continue
            for f in p.iterdir():
                if not f.is_file():
                    continue
                sk = source_key(f.stem)
                if sk not in sk_info:
                    sk_info[sk] = {"class": cls, "splits": defaultdict(int)}
                sk_info[sk]["splits"][split] += 1

    # Patches per source
    patch_counts_all = [sum(v["splits"].values()) for v in sk_info.values()]
    patch_counts_all.sort()
    log(f"  Distinct source keys overall : {len(sk_info)}")
    log(f"  Patches per source  min      : {min(patch_counts_all)}")
    log(f"  Patches per source  median   : {int(np.median(patch_counts_all))}")
    log(f"  Patches per source  max      : {max(patch_counts_all)}")

    # Per-class
    sk_by_class: dict[str, list[str]] = defaultdict(list)
    for sk, info in sk_info.items():
        sk_by_class[info["class"]].append(sk)
    log("")
    for cls in CLASSES:
        keys = sk_by_class[cls]
        cls_counts = [sum(sk_info[k]["splits"].values()) for k in keys]
        if cls_counts:
            log(f"  {cls:<18}: {len(keys)} source keys, patches min/med/max = "
                f"{min(cls_counts)}/{int(np.median(cls_counts))}/{max(cls_counts)}")
        else:
            log(f"  {cls:<18}: 0 source keys")

    # Cross-split leakage
    log("\n  Cross-split appearance:")
    leaking_sources: list[str] = []
    for sk, info in sk_info.items():
        splits_present = list(info["splits"].keys())
        if len(splits_present) > 1:
            leaking_sources.append(sk)

    if leaking_sources:
        log(f"  LEAKAGE: {len(leaking_sources)} source key(s) appear in multiple splits.")
        for sk in sorted(leaking_sources)[:20]:
            splits_str = ", ".join(f"{s}={sk_info[sk]['splits'][s]}" for s in SPLITS if s in sk_info[sk]["splits"])
            log(f"    {sk} [{sk_info[sk]['class']}] : {splits_str}")
        if len(leaking_sources) > 20:
            log(f"    ... and {len(leaking_sources) - 20} more")
    else:
        log("  CLEAN: no source key appears in more than one split.")

    # Full source key table
    log("\n  All source keys (key | class | total patches | splits):")
    log(f"  {'Key':<30} {'Class':<14} {'Patches':>8}  Splits")
    log("  " + "-" * 70)
    for sk in sorted(sk_info):
        info = sk_info[sk]
        total_p = sum(info["splits"].values())
        splits_str = "  ".join(f"{s}:{info['splits'][s]}" for s in SPLITS if s in info["splits"])
        log(f"  {sk:<30} {info['class']:<14} {total_p:>8}  {splits_str}")

    report["source_grouping"] = {
        "distinct_source_keys": len(sk_info),
        "patches_per_source": {"min": min(patch_counts_all), "median": int(np.median(patch_counts_all)), "max": max(patch_counts_all)},
        "leaking_sources": leaking_sources,
        "leakage_detected": len(leaking_sources) > 0,
        "source_keys": {
            sk: {"class": info["class"], "splits": dict(info["splits"]), "total_patches": sum(info["splits"].values())}
            for sk, info in sk_info.items()
        },
    }

    # â”€â”€ 4. IMAGE PROPERTIES (sample) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log(f"4. IMAGE PROPERTIES (random sample up to {SAMPLE_PER_CLASS}/class, seed={SEED})")
    log("â”€" * 72)

    sample_files: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_files = []
        for split in SPLITS:
            p = root / split / cls
            if p.is_dir():
                cls_files.extend(f for f in p.iterdir() if f.is_file())
        rng = random.Random(SEED)
        rng.shuffle(cls_files)
        sample_files[cls] = cls_files[:SAMPLE_PER_CLASS]

    size_counts: dict[str, dict] = {}
    channel_counts: dict[str, dict] = {}
    non_100x100: dict[str, list] = {}
    load_errors_props: list[str] = []

    for cls in CLASSES:
        sizes: dict[tuple, int] = defaultdict(int)
        chans: dict[int, int] = defaultdict(int)
        bad = []
        for fp in sample_files[cls]:
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is None:
                load_errors_props.append(str(fp))
                continue
            h, w = img.shape[:2]
            c = 1 if img.ndim == 2 else img.shape[2]
            sizes[(h, w)] += 1
            chans[c] += 1
            if h != 100 or w != 100:
                bad.append((str(fp.name), h, w))
        size_counts[cls] = dict(sizes)
        channel_counts[cls] = dict(chans)
        non_100x100[cls] = bad

    log("")
    for cls in CLASSES:
        log(f"  [{cls}]")
        log(f"    Unique (h,w) sizes: {dict(sorted(size_counts[cls].items()))}")
        flagged = non_100x100[cls]
        if flagged:
            log(f"    NON-100x100 patches: {len(flagged)} â€” sample: {flagged[:5]}")
        else:
            log(f"    All sampled patches are exactly 100Ã—100.")
        log(f"    Channel count distribution: {dict(sorted(channel_counts[cls].items()))}")
        # dtype
        dtypes = defaultdict(int)
        for fp in sample_files[cls]:
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is not None:
                dtypes[str(img.dtype)] += 1
        log(f"    dtype distribution: {dict(dtypes)}")
        # grayscale check: if 3-channel, are all channels identical?
        if channel_counts[cls].get(3, 0) > 0:
            n_true_gray = 0
            n_checked = 0
            for fp in sample_files[cls][:200]:
                img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 3:
                    n_checked += 1
                    if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,0], img[:,:,2]):
                        n_true_gray += 1
            log(f"    3-channel truly grayscale (checked up to 200): {n_true_gray}/{n_checked}")

    report["image_properties"] = {
        "size_counts": {cls: {str(k): v for k, v in size_counts[cls].items()} for cls in CLASSES},
        "channel_counts": {cls: channel_counts[cls] for cls in CLASSES},
        "non_100x100_count": {cls: len(non_100x100[cls]) for cls in CLASSES},
    }

    # â”€â”€ 5. PIXEL STATISTICS (same sample) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("5. PIXEL STATISTICS (same sample)")
    log("â”€" * 72)

    pix_stats: dict[str, dict] = {}
    for cls in CLASSES:
        means, stds, mins, maxs, zero_fracs = [], [], [], [], []
        for fp in sample_files[cls]:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            arr = img.astype(np.float32)
            means.append(arr.mean())
            stds.append(arr.std())
            mins.append(arr.min())
            maxs.append(arr.max())
            zero_fracs.append((arr == 0).mean())
        if means:
            pix_stats[cls] = {
                "mean": float(np.mean(means)),
                "std": float(np.mean(stds)),
                "min": float(np.min(mins)),
                "max": float(np.max(maxs)),
                "mean_zero_frac": float(np.mean(zero_fracs)),
                "n_images": len(means),
            }
        else:
            pix_stats[cls] = {}

    log("")
    log(f"  {'Class':<18} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}  {'Mean zero-px frac':>20}")
    log("  " + "-" * 72)
    for cls in CLASSES:
        s = pix_stats[cls]
        if s:
            log(f"  {cls:<18} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>6.0f} {s['max']:>6.0f}  {s['mean_zero_frac']:>20.4f}")
        else:
            log(f"  {cls:<18} (no data)")
    log("\n  Note: 'mean zero-px frac' = mean fraction of exactly-zero (black) pixels")
    log("        per image. High value â†’ large masked/background areas in each patch.")

    report["pixel_statistics"] = pix_stats

    # â”€â”€ 6. REDUNDANCY (pixel-hash, same sample) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log(f"6. REDUNDANCY (pixel-hash on grayscale, capped at {SAMPLE_PER_CLASS}/class)")
    log("â”€" * 72)

    redundancy: dict[str, dict] = {}
    for cls in CLASSES:
        hashes: dict[str, list[str]] = defaultdict(list)
        n_loaded = 0
        for fp in sample_files[cls]:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            n_loaded += 1
            hashes[pixel_hash(img)].append(fp.name)
        dup_groups = {h: names for h, names in hashes.items() if len(names) > 1}
        n_dup_files = sum(len(v) - 1 for v in dup_groups.values())  # extras above first
        redundancy[cls] = {
            "n_loaded": n_loaded,
            "unique_pixel_hashes": len(hashes),
            "duplicate_groups": len(dup_groups),
            "duplicate_extra_files": n_dup_files,
            "duplicate_rate_pct": round(100.0 * n_dup_files / n_loaded, 2) if n_loaded else 0,
        }

    log("")
    log(f"  {'Class':<18} {'Loaded':>8} {'Unique':>8} {'Dup groups':>12} {'Dup extras':>12} {'Dup%':>8}")
    log("  " + "-" * 72)
    for cls in CLASSES:
        r = redundancy[cls]
        log(f"  {cls:<18} {r['n_loaded']:>8} {r['unique_pixel_hashes']:>8} "
            f"{r['duplicate_groups']:>12} {r['duplicate_extra_files']:>12} {r['duplicate_rate_pct']:>8.2f}%")

    report["redundancy"] = redundancy

    # â”€â”€ 7. INTEGRITY â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("7. INTEGRITY (all files in sample)")
    log("â”€" * 72)

    bad_files: list[str] = []
    for cls in CLASSES:
        for fp in sample_files[cls]:
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is None:
                bad_files.append(str(fp))

    if bad_files:
        log(f"  {len(bad_files)} file(s) failed to load:")
        for bf in bad_files[:20]:
            log(f"    {bf}")
    else:
        log(f"  All sampled files loaded successfully.")

    report["integrity"] = {"unreadable_files": bad_files}

    # â”€â”€ 8. VISUAL MONTAGE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    log("\n" + "â”€" * 72)
    log("8. VISUAL MONTAGE")
    log("â”€" * 72)

    montage_path = out_dir / "dataset_montage.png"
    N_COLS = 5
    N_ROWS = len(CLASSES)
    CELL   = 120   # cell size in pixels (100px image + margin)
    MARGIN = 10
    LABEL_H = 20

    canvas = np.ones((N_ROWS * (CELL + LABEL_H + MARGIN) + 40, N_COLS * CELL + MARGIN, 3), dtype=np.uint8) * 240

    try:
        rng = random.Random(SEED + 1)
        for row_idx, cls in enumerate(CLASSES):
            all_cls_files = []
            for split in SPLITS:
                p = root / split / cls
                if p.is_dir():
                    all_cls_files.extend(f for f in p.iterdir() if f.is_file())
            rng.shuffle(all_cls_files)
            chosen = all_cls_files[:N_COLS]
            for col_idx, fp in enumerate(chosen):
                img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (100, 100))
                y0 = row_idx * (CELL + LABEL_H + MARGIN) + 30
                x0 = col_idx * CELL + MARGIN
                canvas[y0:y0+100, x0:x0+100] = img_resized
                # class label on first column
                if col_idx == 0:
                    cv2.putText(canvas, cls, (x0, y0 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 150), 1, cv2.LINE_AA)
        cv2.imwrite(str(montage_path), canvas)
        log(f"  Montage saved : {montage_path.resolve()}")
    except Exception as exc:
        log(f"  ERROR generating montage: {exc}")

    report["montage"] = str(montage_path.resolve())

    # â”€â”€ Write outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    txt_path = out_dir / "dataset_report.txt"
    json_path = out_dir / "dataset_report.json"

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    log("\n" + "=" * 72)
    log(f"Reports written:")
    log(f"  {txt_path.resolve()}")
    log(f"  {json_path.resolve()}")
    log(f"  {montage_path.resolve()}")
    log("=" * 72)


if __name__ == "__main__":
    main()

