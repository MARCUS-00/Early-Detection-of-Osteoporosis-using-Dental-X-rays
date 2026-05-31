"""Thin wrapper: build data/manifest.csv from DATASET_ROOT."""
import argparse, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=None)
parser.add_argument("--output", default="data/manifest.csv")
args = parser.parse_args()

if args.dataset_root is None:
    cfg = Path(__file__).parent.parent / "configs" / "classifier.yaml"
    try:
        import yaml
        c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        td = c.get("train_dir", "")
        if not td:
            print("ERROR: pass --dataset_root or set train_dir in configs/classifier.yaml")
            sys.exit(1)
        args.dataset_root = str(Path(td).parent)
    except Exception as e:
        print(f"ERROR: {e}"); sys.exit(1)

from osteo.evaluation.build_manifest import build_manifest
build_manifest(args.dataset_root, args.output)
