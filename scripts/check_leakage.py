"""
Phase C leakage check.

Groups patch filenames by source key = stem with trailing _<digits> stripped.
Reports per-split/class counts, distinct source-key counts, sample keys,
and any source key that appears in more than one split.

Usage:
    python scripts/check_leakage.py --dataset_root "<path to 100x100 folder>"

If no argument is given, reads train_dir from configs/classifier.yaml and
infers the 100x100 root from it (strips the trailing /train component).
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


def source_key(stem: str) -> str:
    """Strip trailing _<digits> suffix to recover the source image identifier."""
    return re.sub(r"_\d+$", "", stem)


def scan_split(split_path: Path) -> dict[str, set[str]]:
    """Return {class_name: set_of_source_keys} for one split folder."""
    result: dict[str, set[str]] = {}
    for class_dir in sorted(split_path.iterdir()):
        if not class_dir.is_dir():
            continue
        keys = {source_key(f.stem) for f in class_dir.iterdir() if f.is_file()}
        result[class_dir.name] = keys
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-image leakage check")
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Path to the 100x100/ folder containing train/valid/test sub-dirs.",
    )
    args = parser.parse_args()

    if args.dataset_root:
        root = Path(args.dataset_root)
    else:
        # Fall back: read from configs/classifier.yaml
        config_path = Path(__file__).parent.parent / "configs" / "classifier.yaml"
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            train_dir = cfg.get("train_dir", "")
            if not train_dir:
                print("ERROR: train_dir is empty in configs/classifier.yaml. Pass --dataset_root.")
                sys.exit(1)
            root = Path(train_dir).parent  # strip the /train component
        except Exception as exc:
            print(f"ERROR reading configs/classifier.yaml: {exc}")
            sys.exit(1)

    print(f"Dataset root : {root}")
    if not root.is_dir():
        print("ERROR: dataset_root does not exist.")
        sys.exit(1)

    splits = ["train", "valid", "test"]
    split_data: dict[str, dict[str, set[str]]] = {}
    for split in splits:
        p = root / split
        if not p.is_dir():
            print(f"WARNING: split folder '{split}' not found at {p}")
            continue
        split_data[split] = scan_split(p)

    print("\n--- Per-split / per-class counts ---")
    print(f"{'Split':<8} {'Class':<18} {'Files (approx)':<18} {'Distinct source keys'}")
    print("-" * 70)
    all_keys_by_split: dict[str, set[str]] = defaultdict(set)
    for split, classes in split_data.items():
        for cls, keys in classes.items():
            # Count actual files for this class/split
            n_files = sum(
                1 for f in (root / split / cls).iterdir() if f.is_file()
            )
            print(f"{split:<8} {cls:<18} {n_files:<18} {len(keys)}")
            all_keys_by_split[split].update(keys)

    print("\n--- Distinct source keys per split (all classes combined) ---")
    for split, keys in all_keys_by_split.items():
        sample = sorted(keys)[:5]
        print(f"  {split:<8}: {len(keys):>5} keys  (sample: {sample})")

    print("\n--- Cross-split overlap check ---")
    leakage_found = False
    split_list = list(all_keys_by_split.items())
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            s1, k1 = split_list[i]
            s2, k2 = split_list[j]
            overlap = k1 & k2
            if overlap:
                leakage_found = True
                sample_overlap = sorted(overlap)[:10]
                print(
                    f"  LEAK: {len(overlap)} source key(s) appear in BOTH '{s1}' AND '{s2}'."
                )
                print(f"        Sample leaking keys: {sample_overlap}")
            else:
                print(f"  OK:   '{s1}' and '{s2}' share 0 source keys.")

    print()
    if leakage_found:
        print(
            "VERDICT: DATA LEAKAGE DETECTED.\n"
            "  One or more source images have patches in multiple splits.\n"
            "  Per-patch metrics will be inflated. The dataset MUST be rebuilt\n"
            "  with source-image-disjoint splits before training or evaluation.\n"
            "  This is a USER decision — do not proceed to training."
        )
        sys.exit(2)
    else:
        print(
            "VERDICT: CLEAN — no source-key overlap detected across splits.\n"
            "  It is safe to proceed to Phase D (training)."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
