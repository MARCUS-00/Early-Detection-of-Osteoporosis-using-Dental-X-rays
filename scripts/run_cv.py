"""Thin wrapper: run source-grouped CV for the osteoporosis classifier.

This script is intended to run on Kaggle GPU (or CPU for smoke-test).
See src/osteo/evaluation/cv_runner.py for the full pipeline.

Typical Kaggle usage:
    python scripts/run_cv.py \
        --dataset_root /kaggle/input/osteo-100x100/100x100 \
        --manifest     /kaggle/input/osteo-100x100/manifest.csv \
        --mode         LOSO \
        --epochs       20 \
        --output_dir   /kaggle/working/outputs
"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", required=True)
parser.add_argument("--manifest",     default="data/manifest.csv")
parser.add_argument("--mode",         default="LOSO", choices=["LOSO", "GROUPKFOLD"])
parser.add_argument("--kfolds",       type=int, default=5)
parser.add_argument("--epochs",       type=int, default=20)
parser.add_argument("--output_dir",   default="outputs")
parser.add_argument("--keep_weights", action="store_true")
args = parser.parse_args()

from osteo.evaluation.cv_runner import run_cv
report = run_cv(
    dataset_root=args.dataset_root,
    manifest_path=args.manifest,
    mode=args.mode,
    n_splits=args.kfolds,
    epochs=args.epochs,
    output_dir=args.output_dir,
    keep_weights=args.keep_weights,
)
print(f"\nCV complete. Report at: {report}")
