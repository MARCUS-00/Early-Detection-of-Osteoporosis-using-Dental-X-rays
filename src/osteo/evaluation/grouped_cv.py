"""
Phase B — Source-grouped cross-validation splitter.

Supports:
  - LOSO  (LeaveOneGroupOut, 13 folds for the 13-source dataset)
  - GROUPKFOLD (GroupKFold with configurable k)

Each fold guarantees that no source_key appears in both the train and eval
DataFrames (hard-assert). This is the anti-leakage requirement.

Usage:
    folds = make_folds(manifest_df, mode="LOSO")
    for fold_idx, (train_df, eval_df, fold_info) in enumerate(folds):
        ...
"""

from __future__ import annotations
from typing import Iterator

import numpy as np


def make_folds(
    manifest_df: "pd.DataFrame",
    mode: str = "LOSO",
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple["pd.DataFrame", "pd.DataFrame", dict]]:
    """
    Generate (train_df, eval_df, fold_info) tuples from the manifest.

    Args:
        manifest_df: DataFrame with at least 'source_key' and 'class' columns.
        mode:        "LOSO" or "GROUPKFOLD".
        n_splits:    only used when mode == "GROUPKFOLD".
        seed:        random seed for GroupKFold shuffle.

    Returns:
        List of (train_df, eval_df, fold_info) tuples.
        fold_info keys: fold_index, mode, held_out_sources, eval_class.

    Raises:
        ValueError:  if mode is unknown.
        AssertionError: if any source_key appears in both train and eval.
    """
    import pandas as pd
    from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

    X = manifest_df.index.values
    groups = manifest_df["source_key"].values

    if mode.upper() == "LOSO":
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(X, groups=groups)
    elif mode.upper() == "GROUPKFOLD":
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, groups=groups)
    else:
        raise ValueError(f"Unknown CV mode '{mode}'. Choose 'LOSO' or 'GROUPKFOLD'.")

    folds = []
    for fold_idx, (train_idx, eval_idx) in enumerate(split_iter):
        train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
        eval_df  = manifest_df.iloc[eval_idx].reset_index(drop=True)

        train_sources = set(train_df["source_key"].unique())
        eval_sources  = set(eval_df["source_key"].unique())
        overlap = train_sources & eval_sources
        assert not overlap, (
            f"Fold {fold_idx}: source_key(s) {overlap} appear in BOTH train and eval. "
            f"This must never happen — check the splitter configuration."
        )

        fold_info = {
            "fold_index": fold_idx,
            "mode": mode.upper(),
            "held_out_sources": sorted(eval_sources),
            "eval_classes": sorted(eval_df["class"].unique()),
            "n_train_patches": len(train_df),
            "n_eval_patches": len(eval_df),
        }
        folds.append((train_df, eval_df, fold_info))

    return folds


def print_fold_summary(folds: list[tuple]) -> None:
    print(f"\nCV summary: {len(folds)} fold(s)")
    print(f"{'Fold':>5}  {'Held-out source(s)':<38}  {'Class':<14}  {'Train pch':>10}  {'Eval pch':>10}")
    print("-" * 82)
    for train_df, eval_df, info in folds:
        sources_str = ", ".join(info["held_out_sources"])
        classes_str = "/".join(info["eval_classes"])
        print(f"{info['fold_index']:>5}  {sources_str:<38}  {classes_str:<14}  "
              f"{info['n_train_patches']:>10,}  {info['n_eval_patches']:>10,}")
