"""Methodology - how NeuroNet MRI is built, trained, and evaluated."""

import streamlit as st

st.set_page_config(page_title="Methodology - NeuroNet MRI", page_icon="💡", layout="wide")
st.title("Methodology")

col1, col2, col3 = st.columns(3)

col1.subheader("Data")
col1.write(
    "MRI images from four classes (glioma, meningioma, pituitary tumor, "
    "no tumor), sourced from a public Kaggle brain-tumor MRI dataset. "
    "The dataset ships with separate `Training/` and `Testing/` folders."
)
col1.markdown("**Split methodology**")
col1.write(
    "`Training/` is split into a training set and a validation set "
    "(stratified, seeded). `Testing/` is never touched until final "
    "evaluation - it is not combined with `Training/` and is not used "
    "for any training or model-selection decision. This is a change "
    "from an earlier version of this project, which combined the two "
    "folders before splitting, so its reported test accuracy did not "
    "reflect a genuine held-out evaluation."
)

col2.subheader("Preprocessing")
col2.write(
    "Every image is resized to 150x150px. The same `preprocess_image` "
    "function (see `src/preprocessing.py`) is used at both training and "
    "inference time, so the two paths cannot silently drift apart."
)
col2.markdown("**Model architecture**")
col2.write(
    "EfficientNetB0, pretrained on ImageNet, used as a feature extractor. "
    "A custom head - global average pooling, dropout, and a 4-way "
    "softmax dense layer - is trained on top for this classification task."
)

col3.subheader("Training")
col3.write(
    "Categorical cross-entropy loss, Adam optimizer. Callbacks: "
    "TensorBoard logging, checkpointing on best validation accuracy, "
    "learning-rate reduction on plateau, and early stopping. All random "
    "seeds (Python, NumPy, TensorFlow) are set from a single "
    "configuration object (`src/config.py`) so a run can be reproduced."
)
col3.markdown("**Evaluation**")
col3.write(
    "Beyond overall accuracy: per-class precision, recall, F1, and "
    "support, plus a confusion matrix. Per-class recall is emphasized "
    "because, for this task, missing a tumor entirely is a more serious "
    "error than confusing two tumor subtypes - a single accuracy number "
    "can hide that difference. See the Results page for current numbers."
)

st.divider()
st.subheader("Known limitations")
st.write(
    "- Evaluated only on a single public dataset; performance on scans "
    "from a different scanner, protocol, or patient population is "
    "unknown.\n"
    "- The model's reported \"confidence\" is a raw softmax probability, "
    "not a calibrated estimate of correctness - see the Classify page.\n"
    "- The current preprocessing pipeline reproduces a BGR-channel-order, "
    "unnormalized input convention inherited from the original OpenCV-based "
    "pipeline (see `src/preprocessing.py` docstring) to remain compatible "
    "with the currently trained weights; a future retrain should move to "
    "a more conventional RGB + normalized pipeline.\n"
    "- This is not a medical device and has not undergone any clinical "
    "validation, regulatory review, or radiologist-adjudicated evaluation."
)

st.divider()
st.subheader("Acknowledgements")
st.write("Data source: Kaggle Brain Tumor Classification (MRI) dataset.")
st.write(
    "With thanks to Dr. Murillo and Teaching Assistant Mahyar Abedi for "
    "mentorship during the CMSE 830 course that this project grew out of."
)
