"""
Phase D — Per-fold train + eval runner.

Designed to run on Kaggle GPU (TF preinstalled, CUDA available).
Can also run on CPU (very slow; for smoke-testing only).

For each LOSO fold:
  1. Splits manifest into train/eval DataFrames (source-disjoint by construction).
  2. Trains MobileNetV2 (same architecture + hyperparameters as original source).
  3. Predicts all eval patches; applies majority vote → one label per source.
  4. Saves per-fold patch predictions to outputs/fold_N_preds.csv.
  5. After all folds, calls aggregate.build_cv_report().

Usage (Kaggle kernel or local):
    python scripts/run_cv.py --dataset_root /kaggle/input/osteo-dataset/100x100 \
                             --manifest /kaggle/input/osteo-dataset/manifest.csv \
                             --mode LOSO --epochs 20 --output_dir /kaggle/working

GUARDRAILS:
  - No accuracy numbers are printed from this file; aggregation is in aggregate.py.
  - Per-fold model weights are deleted after evaluation to save disk space,
    unless --keep_weights is passed.
  - pipeline_full.py is NOT called; it remains a NotImplementedError placeholder.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

SEED = 42
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
LR         = 0.0001
CLASSES    = ["Normal", "Osteopenia", "Osteoporosis"]


def _set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass


def run_cv(
    dataset_root: str,
    manifest_path: str,
    mode: str = "LOSO",
    n_splits: int = 5,
    epochs: int = 20,
    output_dir: str = "outputs",
    keep_weights: bool = False,
) -> Path:
    """
    Run the full CV pipeline and return the path to cv_report.txt.

    Args:
        dataset_root:  path to the 100x100 folder.
        manifest_path: path to manifest.csv (built by build_manifest.py).
        mode:          "LOSO" or "GROUPKFOLD".
        n_splits:      number of folds for GROUPKFOLD.
        epochs:        training epochs per fold.
        output_dir:    directory to write outputs.
        keep_weights:  if True, keep per-fold .h5 files.

    Returns:
        Path to outputs/cv_report.txt.
    """
    import pandas as pd
    import tensorflow as tf

    _set_seeds()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest):,} rows, "
          f"{manifest['source_key'].nunique()} sources")

    # ── Generate folds ────────────────────────────────────────────────────────
    from osteo.evaluation.grouped_cv import make_folds, print_fold_summary
    folds = make_folds(manifest, mode=mode, n_splits=n_splits, seed=SEED)
    print_fold_summary(folds)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    from osteo.evaluation.preprocessing import (
        make_preprocess_fn, make_train_datagen, make_eval_datagen, make_flow_kwargs,
    )
    preprocess_fn = make_preprocess_fn()

    # ── Per-fold training ─────────────────────────────────────────────────────
    all_fold_preds = []  # list of DataFrames: columns [source_key, true_class, pred_class]
    patch_fold_accs = []

    for fold_idx, (train_df, eval_df, fold_info) in enumerate(folds):
        held_sources = fold_info["held_out_sources"]
        print(f"\n{'='*64}")
        print(f"Fold {fold_idx}  held-out: {held_sources}  "
              f"(train={fold_info['n_train_patches']:,} / eval={fold_info['n_eval_patches']:,})")
        print(f"{'='*64}")

        # ── Data generators ──────────────────────────────────────────────────
        train_gen_factory = make_train_datagen(preprocess_fn)
        eval_gen_factory  = make_eval_datagen(preprocess_fn)

        train_kwargs = make_flow_kwargs(
            train_df, dataset_root, train_gen_factory,
            shuffle=True, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED,
        )
        eval_kwargs = make_flow_kwargs(
            eval_df, dataset_root, eval_gen_factory,
            shuffle=False, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=SEED,
        )

        train_gen = train_gen_factory.flow_from_dataframe(**train_kwargs)
        eval_gen  = eval_gen_factory.flow_from_dataframe(**eval_kwargs)

        print(f"Train generator: {train_gen.n} patches in {len(CLASSES)} classes")
        print(f"Eval  generator: {eval_gen.n}  patches in {len(CLASSES)} classes")

        # ── Build model (fresh per fold) ─────────────────────────────────────
        tf.keras.backend.clear_session()
        _set_seeds()

        from osteo.classification.model import build_mobilenetv2
        model = build_mobilenetv2(num_classes=3, img_size=IMG_SIZE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3,
            )
        ]

        # ── Train ────────────────────────────────────────────────────────────
        model.fit(
            train_gen,
            epochs=epochs,
            validation_data=eval_gen,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Predict all eval patches ─────────────────────────────────────────
        eval_gen_pred = eval_gen_factory.flow_from_dataframe(
            **{**eval_kwargs, "shuffle": False, "batch_size": 64},
        )
        preds_prob = model.predict(eval_gen_pred, verbose=0)
        patch_pred_indices = np.argmax(preds_prob, axis=1)
        patch_true_indices = eval_gen_pred.classes

        # Patch-level accuracy for this fold (secondary metric)
        patch_acc = float((patch_pred_indices == patch_true_indices).mean())
        patch_fold_accs.append(patch_acc)
        print(f"Fold {fold_idx} patch accuracy (secondary): {patch_acc:.4f}")

        # ── Per-patch predictions DataFrame ──────────────────────────────────
        eval_df_ordered = eval_df.copy()
        eval_df_ordered = eval_df_ordered.iloc[
            list(range(len(eval_gen_pred.filenames)))
        ].reset_index(drop=True)

        fold_patch_df = pd.DataFrame({
            "fold": fold_idx,
            "source_key": eval_df_ordered["source_key"],
            "true_class": [CLASSES[i] for i in patch_true_indices],
            "pred_class": [CLASSES[i] for i in patch_pred_indices],
            "true_idx": patch_true_indices,
            "pred_idx": patch_pred_indices,
        })

        # Save per-fold patch predictions
        fold_pred_path = out / f"fold_{fold_idx:02d}_patch_preds.csv"
        fold_patch_df.to_csv(fold_pred_path, index=False)
        print(f"Fold {fold_idx} patch preds saved: {fold_pred_path}")

        # ── Majority vote per source ──────────────────────────────────────────
        src_rows = []
        for sk in sorted(fold_patch_df["source_key"].unique()):
            mask = fold_patch_df["source_key"] == sk
            votes = fold_patch_df.loc[mask, "pred_idx"].values
            counts = np.bincount(votes, minlength=len(CLASSES))
            voted_label = int(np.argmax(counts))
            true_label  = int(fold_patch_df.loc[mask, "true_idx"].iloc[0])
            src_rows.append({
                "fold": fold_idx,
                "source_key": sk,
                "true_class": CLASSES[true_label],
                "pred_class": CLASSES[voted_label],
                "correct": CLASSES[true_label] == CLASSES[voted_label],
                "votes": {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
            })
        all_fold_preds.extend(src_rows)

        # ── Optionally discard weights to save disk ───────────────────────────
        weight_path = out / f"fold_{fold_idx:02d}_model.h5"
        if keep_weights:
            model.save(str(weight_path))
            print(f"Weights saved: {weight_path}")
        del model
        tf.keras.backend.clear_session()

    # ── Aggregate and write report ────────────────────────────────────────────
    from osteo.evaluation.aggregate import build_cv_report
    import json

    # Save all-folds source predictions
    src_preds_path = out / "all_source_preds.json"
    src_preds_path.write_text(json.dumps(all_fold_preds, indent=2))

    report_path = build_cv_report(
        source_preds=all_fold_preds,
        patch_fold_accs=patch_fold_accs,
        mode=mode,
        output_dir=str(out),
    )
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run source-grouped CV for osteoporosis classifier")
    parser.add_argument("--dataset_root", required=True, help="Path to 100x100 folder")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--mode", default="LOSO", choices=["LOSO", "GROUPKFOLD"])
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--keep_weights", action="store_true")
    args = parser.parse_args()

    report = run_cv(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        mode=args.mode,
        n_splits=args.kfolds,
        epochs=args.epochs,
        output_dir=args.output_dir,
        keep_weights=args.keep_weights,
    )
    print(f"\nCV complete. Report: {report}")
