# NeuroNet MRI

A convolutional neural network (EfficientNetB0 transfer learning) that
classifies brain MRI scans into glioma, meningioma, pituitary tumor, or
no tumor, served through a Streamlit app.

> **This is a research and educational project, not a medical device.**
> See [Responsible Use](#responsible-use) before doing anything with it
> beyond learning and experimentation.

## Overview

NeuroNet MRI started as a course project (CMSE 830) and has since been
refactored into a more defensible, maintainable application: a corrected
train/validation/test methodology, a clean separation between training
and inference code, reusable preprocessing, input validation, and
honest performance reporting in place of the original's overstated
accuracy claim.

## Demo

The Streamlit app has four pages: **Home**, **Classify** (upload a scan
and get a prediction), **Results** (measured performance, data-driven -
see [Results](#results)), **Methodology**, and **About**.

## Problem

Manually reviewing MRI scans for brain tumors is slow and requires
specialist expertise. This project explores whether a CNN can learn to
distinguish tumor types from MRI slices, as a learning exercise in
applied deep learning and, separately, in building a trustworthy ML
application around a model - not as a substitute for radiological
review.

## Dataset

MRI images from the public Kaggle **Brain Tumor Classification (MRI)**
dataset, with four classes: `glioma_tumor`, `meningioma_tumor`,
`pituitary_tumor`, `no_tumor`. The dataset ships with its own
`Training/` (3,870 images) and `Testing/` (394 images) split, which this
project now respects as a true train/validation vs. held-out-test split
(see [Approach](#approach)).

The dataset is not included in this repository (see `.gitignore`) - to
retrain locally, download it and place it at `mri-data/Training/` and
`mri-data/Testing/` matching the folder names in
`src/config.DATASET_FOLDER_TO_CLASS`. Check the dataset's own license/
usage terms on Kaggle before any use beyond personal experimentation.

## Approach

1. **Preprocessing** - every image resized to 150x150px through the
   single shared `src/preprocessing.py` function, used identically at
   training and inference time (see that file's docstring for an
   important channel-order detail inherited from the original pipeline).
2. **Model** - EfficientNetB0 (ImageNet-pretrained) as a frozen feature
   extractor, with a global-average-pooling + dropout + 4-way softmax
   head trained on top.
3. **Split** - `Training/` is split into train/validation (stratified,
   seeded, 90/10). `Testing/` is held out entirely until final
   evaluation. The original version of this project combined both
   folders before splitting, which meant its "test" accuracy wasn't a
   genuine held-out measurement - see
   `reports/baseline_legacy/README.md` for the full explanation.
4. **Reproducibility** - all seeds (Python, NumPy, TensorFlow) and all
   hyperparameters are set from one place: `src/config.py`.

## Model Architecture

```
EfficientNetB0 (frozen, ImageNet weights, no top)
  -> GlobalAveragePooling2D
  -> Dropout(0.4)
  -> Dense(4, activation="softmax")
```

## Evaluation

Run `python training/evaluate.py` to evaluate the current model against
`mri-data/Testing/` and generate:

- `reports/metrics.json` - accuracy, macro F1, per-class precision/
  recall/F1/support
- `reports/classification_report.csv`
- `reports/confusion_matrix.csv`

Per-class **recall** is emphasized over plain accuracy: for this task, a
tumor scan misclassified as "no tumor" is a materially worse mistake
than confusion between two tumor subtypes, and overall accuracy alone
can hide that.

## Results

See the **Results** page in the running app, or `reports/metrics.json`
directly. This project ships without a retrained model artifact
evaluated under the corrected methodology (see
[Limitations](#limitations) - the refactor was done in an environment
without GPU/TensorFlow access); running `training/train.py` followed by
`training/evaluate.py` will populate real, current numbers.

`reports/baseline_legacy/` preserves the original project's numbers
(~96.9% accuracy under a flawed, leakage-prone split) for historical
reference only - they should not be quoted as a measure of true
held-out performance. The original homepage's claim of 99.8% accuracy
did not match even those numbers and has been removed.

## Limitations

- Evaluated only on one public dataset; performance on scans from a
  different scanner, acquisition protocol, or patient population is
  unknown.
- The Classify page's "confidence" is a raw softmax probability, not a
  calibrated estimate of correctness, and definitely not a clinical
  confidence score.
- This refactor was carried out without access to a GPU or a
  TensorFlow-capable environment, so the corrected training/evaluation
  pipeline (`training/`) has been written and unit-tested at the
  interface level (mocked model), but the model has not actually been
  retrained end-to-end since the split was fixed. Treat
  `reports/baseline_legacy/` as historical only, and run
  `training/train.py` + `training/evaluate.py` yourself to get current,
  trustworthy numbers before relying on this project's accuracy claims.
- The preprocessing pipeline currently reproduces a BGR-channel-order,
  unnormalized input convention inherited from the original code, to
  stay compatible with the legacy model weights (see
  `src/preprocessing.py`). A future retrain should move to a
  conventional RGB + normalized pipeline.
- No calibration, uncertainty estimation, or out-of-distribution
  detection beyond a simple confidence threshold.
- Model interpretability (e.g. Grad-CAM) is not implemented; see
  roadmap Phase 11 for a suggested follow-up.

## Responsible Use

- This is not a medical device and has not been clinically validated
  or reviewed by any regulatory body.
- Do not use this project's predictions to make, or to inform, real
  diagnostic or treatment decisions.
- Uploaded images may be out-of-distribution relative to the training
  data (different scanner, protocol, orientation, or not an MRI at
  all); the app validates that an upload is a well-formed image, but
  makes no claim to reliably detect whether it is actually an MRI scan.

## Project Structure

```
neuronet-mri/
├── app.py                  # Streamlit entry point (home page)
├── pages/
│   ├── classify.py         # upload + prediction UI
│   ├── results.py          # data-driven performance reporting
│   ├── methods.py          # methodology write-up
│   └── about.py            # project/developer background
├── src/
│   ├── config.py           # single source of truth: paths, classes, hyperparams
│   ├── validation.py        # upload validation (extension/size/corruption)
│   ├── preprocessing.py     # shared training+inference preprocessing
│   ├── model.py             # centralized, cached model loading
│   ├── inference.py         # single structured predict() function
│   └── evaluation.py        # metrics computation used by training/evaluate.py
├── training/
│   ├── train.py             # corrected train/validation split, reproducible
│   └── evaluate.py          # the only code path that touches Testing/
├── notebooks/
│   └── exploration.ipynb    # original exploratory notebook, kept for history
├── models/
│   └── neuronet_mri_legacy.keras
├── reports/
│   └── baseline_legacy/      # original (flawed-split) numbers, clearly labeled
├── assets/                  # static images/logos used by the app
├── tests/
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   └── test_inference.py
├── requirements.txt          # app runtime deps only
├── requirements-training.txt # + training/eval deps
├── requirements-dev.txt      # + pytest
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Training

Requires `requirements-training.txt` and a local copy of the dataset at
`mri-data/` (see [Dataset](#dataset)):

```bash
pip install -r requirements.txt -r requirements-training.txt
python training/train.py       # trains + saves models/neuronet_mri.keras
python training/evaluate.py    # evaluates against Testing/, writes reports/
```

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

`test_preprocessing.py` and `test_validation.py` need only PIL/numpy.
`test_inference.py` uses a mock model, so the full suite runs without
TensorFlow installed.

## Technologies

TensorFlow / Keras, EfficientNetB0, Streamlit, NumPy, OpenCV, pandas,
scikit-learn, pytest.

## Future Work

See the full roadmap for the complete list; the highlights not yet done:

- Retrain end-to-end under the corrected split and refresh
  `reports/metrics.json` with real, current numbers.
- Data augmentation, class-imbalance handling, and a simpler baseline
  model for comparison.
- Grad-CAM (or similar) interpretability on the Classify page.
- Calibration / uncertainty estimation beyond a single confidence
  threshold.
- CI, containerization, and experiment tracking.

## Acknowledgements

Data source: Kaggle Brain Tumor Classification (MRI) dataset. With
thanks to Dr. Murillo and Teaching Assistant Mahyar Abedi for mentorship
during the CMSE 830 course this project grew out of.

Developed by Mikayla Norton.
