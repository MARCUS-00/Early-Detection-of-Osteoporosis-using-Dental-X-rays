"""
Phase D -- Per-fold train + eval runner.

Designed to run on Kaggle GPU (TF preinstalled, CUDA available).
Can also run on CPU (very slow; for smoke-testing only).

For each LOSO fold:
  1. Splits manifest into train/eval DataFrames (source-disjoint by construction).
  2. Optionally subsamples each per source_key (see train_sample_per_source).
  3. Trains MobileNetV2 (same architecture + hyperparameters as original source).
  4. Predicts all eval patches; applies majority vote -> one label per source.
  5. Saves per-fold patch predictions to outputs/fold_NN_patch_preds.csv.
  6. If fold_NN_patch_preds.csv already exists, loads and skips training (resume).
  7. After all folds, calls aggregate.build_cv_report().

Usage (Kaggle kernel or local):
    python scripts/run_cv.py --dataset_root /kaggle/input/osteo-dataset/100x100 \\
                             --manifest /kaggle/working/manifest.csv \\
                             --mode LOSO --epochs 20 --output_dir /kaggle/working/outputs

GUARDRAILS:
  - No accuracy numbers are printed from this file; aggregation is in aggregate.py.
  - Per-fold model weights are deleted after evaluation to save disk space,
    unless --keep_weights is passed.
  - pipeline_full.py is NOT called; it remains a NotImplementedError placeholder.
"""

import argparse
import random
from pathlib import Path

import numpy as np

SEED       = 42
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
LR         = 0.0001
CLASSES    = ["Normal", "Osteopenia", "Osteoporosis"]


def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def _subsample_per_source(df, n: int, seed: int):
    """
    Return up to *n* rows per source_key, sampled without replacement.

    Justified: the ~5,775 patches per source are all augmentations of a single
    source image. Subsampling cuts GPU time and equalises per-source counts
    without losing meaningful coverage.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in df.groupby("source_key", sort=True):
        if len(grp) <= n:
            parts.append(grp)
        else:
            idx = rng.choice(len(grp), size=n, replace=False)
            parts.append(grp.iloc[sorted(idx)])
    return pd.concat(parts).reset_index(drop=True)


def _reconstruct_source_preds(fold_patch_df, fold_idx: int):
    """
    From a per-patch predictions DataFrame return:
      - list of per-source majority-vote dicts (compatible with aggregate.py)
      - patch-level accuracy for this fold (float)

    Used for both fresh results and resumed-from-CSV results.
    """
    patch_acc = float(
        (fold_patch_df["true_idx"].astype(int) == fold_patch_df["pred_idx"].astype(int)).mean()
    )

    src_rows = []
    for sk in sorted(fold_patch_df["source_key"].unique()):
        mask  = fold_patch_df["source_key"] == sk
        votes = fold_patch_df.loc[mask, "pred_idx"].astype(int).values
        counts = np.bincount(votes, minlength=len(CLASSES))
        voted_label = int(np.argmax(counts))
        true_label  = int(fold_patch_df.loc[mask, "true_idx"].astype(int).iloc[0])
        src_rows.append({
            "fold":       fold_idx,
            "source_key": sk,
            "true_class": CLASSES[true_label],
            "pred_class": CLASSES[voted_label],
            "correct":    CLASSES[true_label] == CLASSES[voted_label],
            "votes":      {CLASSES[i]: int(counts[i]) for i in range(len(CLASSES))},
        })
    return src_rows, patch_acc


def run_cv(
    dataset_root: str,
    manifest_path: str,
    mode: str = "LOSO",
    n_splits: int = 5,
    epochs: int = 20,
    output_dir: str = "outputs",
    keep_weights: bool = False,
    train_sample_per_source=None,
    eval_sample_per_source=None,
    seed: int = SEED,
) -> Path:
    """
    Run the full CV pipeline and return the path to cv_report.txt.

    Args:
        dataset_root:             path to the 100x100 folder.
        manifest_path:            path to manifest.csv (built by build_manifest.py).
        mode:                     "LOSO" or "GROUPKFOLD".
        n_splits:                 number of folds for GROUPKFOLD.
        epochs:                   training epochs per fold.
        output_dir:               directory to write outputs.
        keep_weights:             if True, keep per-fold .h5 files.
        train_sample_per_source:  int or None -- max patches per source in train set.
        eval_sample_per_source:   int or None -- max patches per source in eval set.
        seed:                     random seed for subsampling and training.

    Returns:
        Path to outputs/cv_report.txt.
    """
    import pandas as pd
    import tensorflow as tf

    _set_seeds(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Load manifest ----------------------------------------------------------------
    manifest = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(manifest):,} rows, "
          f"{manifest['source_key'].nunique()} sources")

    # -- Generate folds ---------------------------------------------------------------
    from osteo.evaluation.grouped_cv import make_folds, print_fold_summary
    folds = make_folds(manifest, mode=mode, n_splits=n_splits, seed=seed)
    print_fold_summary(folds)

    if train_sample_per_source is not None or eval_sample_per_source is not None:
        print(f"Per-source subsampling: "
              f"train<={train_sample_per_source or 'all'}, "
              f"eval<={eval_sample_per_source or 'all'}")

    # -- Preprocessing ----------------------------------------------------------------
    from osteo.evaluation.preprocessing import (
        make_preprocess_fn, make_train_datagen, make_eval_datagen, make_flow_kwargs,
    )
    preprocess_fn = make_preprocess_fn()

    # -- Per-fold training ------------------------------------------------------------
    all_fold_preds  = []
    patch_fold_accs = []

    for fold_idx, (train_df, eval_df, fold_info) in enumerate(folds):
        held_sources = fold_info["held_out_sources"]

        # -- Resume: reuse completed fold if its CSV already exists -------------------
        fold_pred_path = out / f"fold_{fold_idx:02d}_patch_preds.csv"
        if fold_pred_path.exists():
            print(f"\nFold {fold_idx} ({held_sources}): "
                  f"resuming from {fold_pred_path.name}")
            loaded = pd.read_csv(fold_pred_path)
            src_rows, patch_acc = _reconstruct_source_preds(loaded, fold_idx)
            all_fold_preds.extend(src_rows)
            patch_fold_accs.append(patch_acc)
            print(f"  Loaded {len(loaded):,} patch preds  "
                  f"patch_acc={patch_acc:.4f}  "
                  f"vote={[r['pred_class'] for r in src_rows]}")
            continue

        print(f"\n{'='*64}")
        print(f"Fold {fold_idx}  held-out: {held_sources}  "
              f"(train={fold_info['n_train_patches']:,} / eval={fold_info['n_eval_patches']:,})")
        print(f"{'='*64}")

        # -- Per-source subsampling ---------------------------------------------------
        train_use = train_df
        eval_use  = eval_df
        if train_sample_per_source is not None:
            train_use = _subsample_per_source(train_df, train_sample_per_source, seed)
            print(f"  Train after sampling: {len(train_use):,} "
                  f"(from {len(train_df):,})")
        if eval_sample_per_source is not None:
            eval_use = _subsample_per_source(eval_df, eval_sample_per_source, seed)
            print(f"  Eval  after sampling: {len(eval_use):,} "
                  f"(from {len(eval_df):,})")

        # -- Data generators ----------------------------------------------------------
        train_gen_factory = make_train_datagen(preprocess_fn)
        eval_gen_factory  = make_eval_datagen(preprocess_fn)

        train_kwargs = make_flow_kwargs(
            train_use, dataset_root, train_gen_factory,
            shuffle=True, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=seed,
        )
        eval_kwargs = make_flow_kwargs(
            eval_use, dataset_root, eval_gen_factory,
            shuffle=False, batch_size=BATCH_SIZE, img_size=IMG_SIZE, seed=seed,
        )

        train_gen = train_gen_factory.flow_from_dataframe(**train_kwargs)
        eval_gen  = eval_gen_factory.flow_from_dataframe(**eval_kwargs)

        print(f"Train generator: {train_gen.n} patches")
        print(f"Eval  generator: {eval_gen.n} patches")

        # -- Build model (fresh per fold) ---------------------------------------------
        tf.keras.backend.clear_session()
        _set_seeds(seed)

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

        # -- Train --------------------------------------------------------------------
        model.fit(
            train_gen,
            epochs=epochs,
            validation_data=eval_gen,
            callbacks=callbacks,
            verbose=1,
        )

        # -- Predict all eval patches -------------------------------------------------
        eval_gen_pred = eval_gen_factory.flow_from_dataframe(
            **{**eval_kwargs, "shuffle": False, "batch_size": 64},
        )
        preds_prob         = model.predict(eval_gen_pred, verbose=0)
        patch_pred_indices = np.argmax(preds_prob, axis=1)
        patch_true_indices = eval_gen_pred.classes

        # -- Build per-patch DataFrame ------------------------------------------------
        fold_patch_df = pd.DataFrame({
            "fold":       fold_idx,
            "source_key": eval_use["source_key"].values[:len(patch_true_indices)],
            "true_class": [CLASSES[i] for i in patch_true_indices],
            "pred_class": [CLASSES[i] for i in patch_pred_indices],
            "true_idx":   patch_true_indices,
            "pred_idx":   patch_pred_indices,
        })
        fold_patch_df.to_csv(fold_pred_path, index=False)
        print(f"Fold {fold_idx} patch preds saved: {fold_pred_path}")

        # -- Majority vote + patch accuracy -------------------------------------------
        src_rows, patch_acc = _reconstruct_source_preds(fold_patch_df, fold_idx)
        all_fold_preds.extend(src_rows)
        patch_fold_accs.append(patch_acc)
        print(f"Fold {fold_idx} patch accuracy (secondary): {patch_acc:.4f}")

        # -- Optionally keep weights --------------------------------------------------
        if keep_weights:
            weight_path = out / f"fold_{fold_idx:02d}_model.h5"
            model.save(str(weight_path))
            print(f"Weights saved: {weight_path}")
        del model
        tf.keras.backend.clear_session()

    # -- Aggregate and write report --------------------------------------------------
    from osteo.evaluation.aggregate import build_cv_report
    import json

    (out / "all_source_preds.json").write_text(json.dumps(all_fold_preds, indent=2))

    report_path = build_cv_report(
        source_preds=all_fold_preds,
        patch_fold_accs=patch_fold_accs,
        mode=mode,
        output_dir=str(out),
    )
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run source-grouped CV for osteoporosis classifier"
    )
    parser.add_argument("--dataset_root", required=True, help="Path to 100x100 folder")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--mode", default="LOSO", choices=["LOSO", "GROUPKFOLD"])
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--keep_weights", action="store_true")
    parser.add_argument("--train_sample", type=int, default=None,
                        help="Max patches per source for training (default: all)")
    parser.add_argument("--eval_sample", type=int, default=None,
                        help="Max patches per source for eval (default: all)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    report = run_cv(
        dataset_root            = args.dataset_root,
        manifest_path           = args.manifest,
        mode                    = args.mode,
        n_splits                = args.kfolds,
        epochs                  = args.epochs,
        output_dir              = args.output_dir,
        keep_weights            = args.keep_weights,
        train_sample_per_source = args.train_sample,
        eval_sample_per_source  = args.eval_sample,
        seed                    = args.seed,
    )
    print(f"\nCV complete. Report: {report}")
