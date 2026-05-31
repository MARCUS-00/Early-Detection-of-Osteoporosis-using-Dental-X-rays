"""
Phase A — Dataset manifest builder.

Scans DATASET_ROOT/{train,valid,test}/{Normal,Osteopenia,Osteoporosis},
records one row per image file, and writes data/manifest.csv.

Stored columns:
    relative_path  – path relative to DATASET_ROOT (portable: survives copy to Kaggle)
    class          – parent folder name (Normal / Osteopenia / Osteoporosis)
    source_key     – filename stem with trailing _<digits> stripped
    split          – "train" / "valid" / "test" (informational only; not used for CV)

The CV folds are derived purely from source_key groups, NOT from the
train/valid/test folders (which have patch-level leakage — see PIPELINE_AUDIT.md).

Asserts: exactly 13 distinct source_keys must be found; stops with RuntimeError otherwise.
"""

import re
import sys
from pathlib import Path

SPLITS  = ["train", "valid", "test"]
CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]
EXPECTED_N_SOURCES = 13


def _source_key(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def build_manifest(dataset_root: str | Path, output_path: str | Path = "data/manifest.csv") -> "pd.DataFrame":
    """
    Scan dataset_root and write manifest CSV.

    Args:
        dataset_root: path to the 100x100 folder (contains train/valid/test).
        output_path:  where to write the CSV (gitignored).

    Returns:
        The manifest DataFrame.

    Raises:
        RuntimeError: if the number of distinct source_keys != 13.
    """
    import pandas as pd

    root = Path(dataset_root)
    rows = []
    for split in SPLITS:
        for cls in CLASSES:
            cls_dir = root / split / cls
            if not cls_dir.is_dir():
                print(f"WARNING: missing folder {cls_dir}", file=sys.stderr)
                continue
            for f in sorted(cls_dir.iterdir()):
                if not f.is_file():
                    continue
                rows.append({
                    "relative_path": f"{split}/{cls}/{f.name}",
                    "class": cls,
                    "source_key": _source_key(f.stem),
                    "split": split,
                })

    if not rows:
        raise RuntimeError(f"No files found under {root}")

    df = pd.DataFrame(rows)
    n_sources = df["source_key"].nunique()
    if n_sources != EXPECTED_N_SOURCES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_N_SOURCES} distinct source_keys; "
            f"found {n_sources}. "
            f"Found keys: {sorted(df['source_key'].unique())}"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Manifest written: {out.resolve()} ({len(df):,} rows, {n_sources} sources)")
    _print_summary(df)
    return df


def _print_summary(df: "pd.DataFrame") -> None:
    print(f"\nSource key summary ({df['source_key'].nunique()} sources):")
    grp = df.groupby(["source_key", "class"]).size().reset_index(name="patches")
    print(grp.to_string(index=False))
    print(f"\nClass totals:")
    print(df["class"].value_counts().to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build dataset manifest")
    parser.add_argument("--dataset_root", default=None, help="Path to 100x100 folder")
    parser.add_argument("--output", default="data/manifest.csv")
    args = parser.parse_args()

    if args.dataset_root is None:
        cfg_path = Path(__file__).parents[3] / "configs" / "classifier.yaml"
        try:
            import yaml
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            train_dir = cfg.get("train_dir", "")
            if not train_dir:
                print("ERROR: pass --dataset_root or set train_dir in classifier.yaml")
                sys.exit(1)
            dataset_root = str(Path(train_dir).parent)
        except Exception as exc:
            print(f"ERROR reading classifier.yaml: {exc}")
            sys.exit(1)
    else:
        dataset_root = args.dataset_root

    build_manifest(dataset_root, args.output)
