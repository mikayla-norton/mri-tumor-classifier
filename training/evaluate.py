"""Evaluate a trained model against the true, untouched held-out test set.

Phase 1.1 / 1.3: `mri-data/Testing/` must never be used for anything
except this final evaluation - not training, not validation, not
hyperparameter selection. This script is the only code path in the
project that reads Testing/, which makes that guarantee easy to audit.

Usage:
    python training/evaluate.py                # evaluates config.MODEL_PATH
    python training/evaluate.py --legacy        # evaluates the legacy model
                                                  (for comparison only - see
                                                  reports/baseline_legacy/README.md)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.evaluation import write_reports
from src.model import load_model
from src.preprocessing import preprocess_batch
from training.train import load_image_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Evaluate the legacy frozen model instead of the current one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write metrics.json / classification_report.csv / "
        "confusion_matrix.csv. Defaults to reports/.",
    )
    args = parser.parse_args()

    model_path = config.LEGACY_MODEL_PATH if args.legacy else config.MODEL_PATH
    output_dir = args.output_dir or config.REPORTS_DIR
    model_version = "legacy_mixed_split" if args.legacy else "current"

    print(f"Loading model from {model_path} ...")
    model = load_model(model_path)

    print("Loading Testing/ (the true held-out set) ...")
    test_images, test_labels = load_image_pool(config.TEST_DATA_DIR)
    print(f"  {len(test_images)} images loaded from Testing/")

    X_test = preprocess_batch(test_images)
    predictions = model.predict(X_test, verbose=1)
    y_pred = predictions.argmax(axis=1).tolist()

    write_reports(
        y_true=test_labels,
        y_pred=y_pred,
        output_dir=output_dir,
        dataset_split="mri-data/Testing (held out, never used in training)",
        model_version=model_version,
    )
    print(f"Reports written to {output_dir}/")
    print(
        "Remember: these numbers describe performance on this public "
        "dataset only. See README > Limitations before quoting them "
        "anywhere as a measure of real-world or clinical performance."
    )


if __name__ == "__main__":
    main()
