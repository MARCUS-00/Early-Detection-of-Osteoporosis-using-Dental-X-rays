# NOT VERIFIED FROM AVAILABLE FILES
"""
Dataset construction placeholder — NOT VERIFIED FROM AVAILABLE FILES.

The original code that:
  - assigned class labels (Normal / Osteopenia / Osteoporosis) to source images,
  - extracted 100×100 patches,
  - and split them into train / valid / test folders

is ABSENT from all source files. Only the resulting on-disk layout has been
confirmed from dataset screenshots.

This module is a PLACEHOLDER describing the expected output layout and the
leakage-safe split requirement. All functions raise NotImplementedError.
"""


def build_100x100_dataset(
    source_root: str,
    output_root: str,
    split: tuple = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> None:
    # NOT VERIFIED FROM AVAILABLE FILES
    """
    Build the 100x100 patch dataset from labelled source images.

    Expected output layout (VERIFIED from dataset screenshots):

        <output_root>/100x100/
            train/
                Normal/
                Osteopenia/
                Osteoporosis/
            valid/              # "valid", NOT "val"
                Normal/
                Osteopenia/
                Osteoporosis/
            test/
                Normal/
                Osteopenia/
                Osteoporosis/

    NOT VERIFIED: The original construction/labelling/splitting code is absent
    from all source artifacts. The layout above is the target; this function
    is a placeholder only.

    CRITICAL — split at the SOURCE-IMAGE level:
        Patch filenames share a source-image prefix (e.g. `roiant5_1oa_*`).
        Any real implementation MUST group all patches from the same source
        image into a single split (train, valid, or test). Splitting
        patch-by-patch without this grouping allows the same source image's
        patches to appear in both train and test, inflating evaluation metrics.

    Args:
        source_root: root directory containing labelled source images.
        output_root: directory where the 100x100/{train,valid,test}/… tree
                     will be written.
        split: (train_fraction, valid_fraction, test_fraction); must sum to 1.
               Claimed value in documentation: (0.70, 0.15, 0.15) — unverified.
        seed: random seed for reproducible splitting.

    Raises:
        NotImplementedError: always — this function is not implemented.
    """
    raise NotImplementedError(
        "Dataset construction code not present in source artifacts. "
        "See module docstring for the required output layout and the "
        "source-image-level split requirement."
    )
