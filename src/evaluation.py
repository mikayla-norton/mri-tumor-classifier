"""Metric computation used by the offline training/evaluation pipeline.

Phase 1.3: beyond accuracy, we compute per-class precision/recall/F1,
macro F1, support, and a confusion matrix. Per-class *recall* is called
out specifically because, for this application, a false negative (a
tumor class misclassified as `no_tumor`) is a materially different kind
of mistake than confusion between two tumor subtypes - plain accuracy
hides that distinction.

This module is intentionally decoupled from Streamlit: it's called by
`training/evaluate.py` to produce the JSON/CSV artifacts that
`pages/results.py` later reads and displays (Phase 6 - the app page
should be data-driven, not a place where metrics get computed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src import config


def compute_classification_report(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> pd.DataFrame:
    """Per-class precision/recall/F1/support plus accuracy and macro/weighted
    averages, as a DataFrame indexed by class name (plus summary rows)."""

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=list(config.CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report_dict).transpose()


def compute_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows=true, cols=predicted)."""

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(config.CLASS_NAMES))))
    return pd.DataFrame(matrix, index=list(config.CLASS_NAMES), columns=list(config.CLASS_NAMES))


def build_metrics_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    dataset_split: str,
    model_version: str,
    training_config: dict | None = None,
) -> dict:
    """Assemble the single metrics.json summary the Results page reads.

    Keeping this as one JSON blob (rather than only CSVs) makes it cheap
    for the Streamlit page to show headline numbers without re-parsing a
    classification-report CSV's slightly awkward shape.
    """

    report_df = compute_classification_report(y_true, y_pred)
    accuracy = float(report_df.loc["accuracy"].iloc[0])
    macro_f1 = float(report_df.loc["macro avg", "f1-score"])

    per_class = {}
    for class_name in config.CLASS_NAMES:
        row = report_df.loc[class_name]
        per_class[class_name] = {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1-score"]),
            "support": int(row["support"]),
        }

    return {
        "dataset_split": dataset_split,
        "model_version": model_version,
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "training_config": training_config or {},
    }


def write_reports(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    output_dir: Path,
    dataset_split: str,
    model_version: str,
    training_config: dict | None = None,
) -> None:
    """Write classification_report.csv, confusion_matrix.csv, and
    metrics.json into `output_dir`, creating it if necessary."""

    output_dir.mkdir(parents=True, exist_ok=True)

    compute_classification_report(y_true, y_pred).to_csv(
        output_dir / "classification_report.csv"
    )
    compute_confusion_matrix(y_true, y_pred).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    metrics = build_metrics_summary(
        y_true,
        y_pred,
        dataset_split=dataset_split,
        model_version=model_version,
        training_config=training_config,
    )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
