# Dataset

The dataset is NOT shipped with this repository (it is large; only its structure is documented here).

## Verified layout (classifier input)

```
100x100/
├── train/
│   ├── Normal/        # grayscale ~100x100 patch images
│   ├── Osteopenia/
│   └── Osteoporosis/
├── valid/             # note: "valid", NOT "val"
│   ├── Normal/
│   ├── Osteopenia/
│   └── Osteoporosis/
└── test/
    ├── Normal/
    ├── Osteopenia/
    └── Osteoporosis/
```

Filenames look like `roiant5_1oa_5199`. Class index order (alphabetical, from
`flow_from_directory`): Normal=0, Osteopenia=1, Osteoporosis=2.

`train.py` consumes `train` + `valid`; `evaluate.py` consumes `test`.

## NOT VERIFIED FROM AVAILABLE FILES

- The code that assigned class labels and split images into train/valid/test is not present in the sources.
- The split ratio (documentation claims 70/15/15) and the split method are unverified.

## WARNING — possible data leakage

Filenames share a common source prefix (e.g. `roiant5_1oa_*`), indicating many patches per source
image. It is NOT confirmed that all patches from a given source stay within a single split.
If the same source's patches appear in more than one split, evaluation metrics are inflated.
Verify per-prefix split-disjointness before trusting any reported accuracy.
