"""Central configuration for NeuroNet MRI.

This is the single source of truth for paths, class labels, image
preprocessing parameters, and training hyperparameters. Both the
training pipeline (``training/``) and the inference application
(``app.py`` / ``pages/``) import from this module so the two can never
silently drift apart.

Nothing in here should require TensorFlow to import, so this module is
safe to import from lightweight contexts (tests, docs generation) too.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "mri-data"
TRAIN_DATA_DIR = DATA_DIR / "Training"
TEST_DATA_DIR = DATA_DIR / "Testing"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "neuronet_mri.keras"
# Legacy artifact kept for provenance. See reports/baseline_legacy/README.md
# for why it must not be presented as the current, trustworthy evaluation.
LEGACY_MODEL_PATH = MODELS_DIR / "neuronet_mri_legacy.keras"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.csv"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.csv"

LOGS_DIR = PROJECT_ROOT / "logs"

# --------------------------------------------------------------------------
# Class labels
# --------------------------------------------------------------------------

# Canonical class order. This order MUST match the order used to build the
# one-hot / integer labels during training, since the model's output index
# `i` is only meaningful relative to this list.
CLASS_NAMES: tuple[str, ...] = (
    "glioma",
    "no_tumor",
    "meningioma",
    "pituitary",
)

# Human-friendly display labels, keyed by the canonical class name above.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "glioma": "Glioma tumor",
    "no_tumor": "No tumor",
    "meningioma": "Meningioma tumor",
    "pituitary": "Pituitary tumor",
}

# Maps the *legacy* dataset folder names (as they exist on disk today) to
# the canonical class names above. Kept separate so a future dataset
# reorganization only requires updating this dict.
DATASET_FOLDER_TO_CLASS: dict[str, str] = {
    "glioma_tumor": "glioma",
    "no_tumor": "no_tumor",
    "meningioma_tumor": "meningioma",
    "pituitary_tumor": "pituitary",
}

# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

IMAGE_SIZE: int = 150  # pixels, square (matches the trained model's input)
IMAGE_CHANNELS: int = 3

# Upload validation (Phase 5)
ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_SIZE_MB: int = 10
MIN_IMAGE_DIMENSION: int = 32  # reject implausibly tiny images

# --------------------------------------------------------------------------
# Confidence / uncertainty handling (Phase 5)
# --------------------------------------------------------------------------

# NOTE: This threshold is a placeholder until it is actually selected using
# validation-set data (see training/evaluate.py and reports/metrics.json).
# It intentionally does NOT claim to represent "diagnostic confidence" -
# it is only the softmax probability of the top predicted class, which is
# a property of the model's training distribution, not a calibrated
# real-world probability of correctness.
UNCERTAIN_CONFIDENCE_THRESHOLD: float = 0.70

# --------------------------------------------------------------------------
# Reproducibility / training hyperparameters (Phase 1.2)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """All knobs that affect a training run, recorded in one place.

    Anything that could change the resulting weights or the evaluation
    numbers should live here so a run is fully described by this object
    (see training/train.py, which serializes this alongside the model).
    """

    seed: int = 101
    image_size: int = IMAGE_SIZE
    batch_size: int = 32
    epochs: int = 15
    validation_split: float = 0.1  # carved out of Training/, never Testing/
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    dropout_rate: float = 0.4
    base_model: str = "EfficientNetB0"
    loss: str = "categorical_crossentropy"

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "base_model": self.base_model,
            "loss": self.loss,
        }


DEFAULT_TRAINING_CONFIG = TrainingConfig()


def set_global_seeds(seed: int = DEFAULT_TRAINING_CONFIG.seed) -> None:
    """Seed every source of randomness we know about.

    Call this once, at the very start of any training or evaluation
    script, before importing/using numpy-random-dependent libraries such
    as scikit-learn's train_test_split.
    """

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        # Determinism ops aren't free, but for a dataset this size the
        # cost is worth defensible, reproducible results.
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    except ImportError:
        # TensorFlow isn't required for modules that only need seeded
        # numpy/random behavior (e.g. computing a train/validation split).
        pass
