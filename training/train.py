"""Train the NeuroNet MRI classifier.

Fixes Phase 1.1 of the roadmap: the original notebook combined
`mri-data/Training` and `mri-data/Testing` and re-split the union with
`train_test_split`, which meant the shipped "test" numbers were not a
true held-out evaluation - some of what had been the original test set
leaked into training, and vice versa.

This script:
  * loads `Training/` and `Testing/` as two separate pools,
  * splits `Training/` into train/validation only (Testing/ is never
    touched until `training/evaluate.py` runs, as a final check),
  * seeds every source of randomness in one place (config.set_global_seeds),
  * uses the single shared `src.preprocessing.preprocess_batch` function
    so training-time preprocessing can never drift from inference-time
    preprocessing,
  * records the full training configuration and history to disk.

Usage:
    python training/train.py

Requires TensorFlow, which is intentionally NOT a dependency of the
Streamlit app itself (see requirements.txt vs requirements-training.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python training/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config
from src.preprocessing import preprocess_batch


def load_image_pool(data_dir: Path) -> tuple[list[Image.Image], list[int]]:
    """Load every image under `data_dir/<folder_name>/*` into memory.

    Returns parallel lists of decoded PIL images and integer class
    indices (indices into config.CLASS_NAMES).
    """

    images: list[Image.Image] = []
    labels: list[int] = []

    for folder_name, class_name in config.DATASET_FOLDER_TO_CLASS.items():
        class_index = config.CLASS_NAMES.index(class_name)
        folder = data_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected dataset folder not found: {folder}. Check "
                f"config.DATASET_FOLDER_TO_CLASS against your data layout."
            )
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                image = Image.open(image_path)
                image.load()
            except (OSError, ValueError):
                print(f"  Skipping unreadable image: {image_path}")
                continue
            images.append(image)
            labels.append(class_index)

    return images, labels


def build_model(training_config: config.TrainingConfig):
    import tensorflow as tf
    from keras.applications import EfficientNetB0

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(training_config.image_size, training_config.image_size, 3),
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=training_config.dropout_rate)(x)
    x = tf.keras.layers.Dense(len(config.CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    model.compile(
        loss=training_config.loss,
        optimizer=training_config.optimizer,
        metrics=["accuracy"],
    )
    return model


def one_hot(labels: list[int]) -> np.ndarray:
    matrix = np.zeros((len(labels), len(config.CLASS_NAMES)), dtype=int)
    for row, label in enumerate(labels):
        matrix[row, label] = 1
    return matrix


def main() -> None:
    training_config = config.DEFAULT_TRAINING_CONFIG
    config.set_global_seeds(training_config.seed)

    import tensorflow as tf
    from keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
        TensorBoard,
    )

    print(f"Training config: {training_config.as_dict()}")

    print("Loading Training/ pool (will be split into train/validation only)...")
    train_pool_images, train_pool_labels = load_image_pool(config.TRAIN_DATA_DIR)
    print(f"  {len(train_pool_images)} images loaded from Training/")

    # NOTE: Testing/ is intentionally not loaded here. It must stay
    # completely unseen until training/evaluate.py runs, or the reported
    # test metrics are meaningless. See module docstring.

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_pool_images,
        train_pool_labels,
        test_size=training_config.validation_split,
        random_state=training_config.seed,
        stratify=train_pool_labels,
    )
    print(f"  train={len(train_images)}  validation={len(val_images)}")

    X_train = preprocess_batch(train_images, image_size=training_config.image_size)
    X_val = preprocess_batch(val_images, image_size=training_config.image_size)
    y_train = one_hot(train_labels)
    y_val = one_hot(val_labels)

    model = build_model(training_config)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        TensorBoard(log_dir=str(config.LOGS_DIR)),
        ModelCheckpoint(
            str(config.MODEL_PATH), monitor="val_accuracy", save_best_only=True, mode="max"
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.3, patience=2, min_delta=0.001, mode="max"
        ),
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        CSVLogger(str(config.TRAINING_HISTORY_PATH)),
    ]

    start = time.time()
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.1f} minutes.")

    # ModelCheckpoint already saved the best-val-accuracy weights to
    # config.MODEL_PATH, but save again explicitly in case training
    # finished without a checkpointed improvement (e.g. epochs=1 runs).
    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    run_metadata = {
        "training_config": training_config.as_dict(),
        "n_train": len(train_images),
        "n_validation": len(val_images),
        "tensorflow_version": tf.__version__,
    }
    with open(config.REPORTS_DIR / "training_run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(
        "Done. Run `python training/evaluate.py` to evaluate this model "
        "against the held-out Testing/ set and generate reports/."
    )


if __name__ == "__main__":
    main()
