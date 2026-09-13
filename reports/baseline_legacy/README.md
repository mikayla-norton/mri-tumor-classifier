# Legacy baseline (Phase 0.1 frozen artifact)

This folder preserves exactly what the original project produced, before
the Phase 1 refactor, for provenance and comparison purposes only.

## What's here

- `classification_report.csv` - the original per-class precision/recall/F1
- `confusion_matrix.csv` - the original confusion matrix
- `training_accuracy_loss.png` - the original training curves
- `confusion_matrix.png` - the original confusion matrix plot

## Why these numbers should not be trusted as-is

The original notebook (`notebooks/exploration.ipynb`) loaded images from
both `mri-data/Training/` **and** `mri-data/Testing/`, concatenated them,
and then ran a fresh `train_test_split` on the combined pool:

```python
X_train, X_test, y_train, y_test = train_test_split(
    train_img, train_labels, test_size=0.1, random_state=101
)
```

That means the resulting "test set" (`X_test`) was not the original,
untouched `Testing/` folder - it was a random 10% slice of the combined
pool, which could include images the model had already seen a near-
duplicate of during training, and which no longer represents a genuine
held-out evaluation. The measured ~96.9% accuracy above is real, in the
sense that it's an honest report of what that (flawed) procedure
produced, but it does **not** tell you how the model performs on truly
unseen data the way a correct train/validation/test split would.

The original project's homepage additionally claimed **99.8% accuracy**,
which does not match even this flawed evaluation's ~96.9% - that claim
has been removed from the app (see `app.py` and the project README).

## What replaced this

`training/train.py` and `training/evaluate.py` implement a corrected
split: `Training/` is split into train/validation only, and `Testing/`
is used exactly once, at the end, purely for evaluation. Once you run
those scripts, current results land in `reports/metrics.json`,
`reports/classification_report.csv`, and `reports/confusion_matrix.csv`
(one level up from this folder) and are what the Results page displays
by default.

This legacy folder is kept read-only, for the record - not overwritten,
so anyone auditing the project's history can see exactly what changed
and why.
