"""NeuroNet MRI - Streamlit entry point.

This file only renders the home page. All ML logic (preprocessing,
inference, model loading) lives in `src/` and is never imported here
directly - see `pages/classify.py` for where it's actually used.

Design note: this page intentionally contains no decorative photography
or illustration. The original version used AI-generated stock imagery
(glowing sci-fi scanners, a fabricated doctor) that added visual noise
without adding information, and sat awkwardly next to a project whose
whole point is being straightforward about what the model actually does.
The only images on this page are real: framework logos and, elsewhere in
the app, actual MRI figures / a real headshot.
"""

import streamlit as st
from PIL import Image

from src import config

st.set_page_config(page_title="NeuroNet MRI", page_icon=":brain:", layout="wide")

st.title("NeuroNet MRI")
st.subheader(
    "A research demo that classifies brain MRI scans by tumor type, "
    "built with a convolutional neural network."
)

st.warning(
    "**This is a research and educational project, not a medical device.** "
    "It is not validated for clinical use, has not been reviewed by any "
    "regulatory body, and must not be used to inform real diagnostic or "
    "treatment decisions. See the Methodology page for full details and "
    "the Results page for measured performance.",
    icon="⚠️",
)

st.divider()

# --------------------------------------------------------------------------
# What it does
# --------------------------------------------------------------------------

st.subheader("What it classifies")
st.write(
    "Given an uploaded MRI scan, the model predicts one of four classes "
    "and reports its confidence for each:"
)

class_columns = st.columns(len(config.CLASS_NAMES))
class_descriptions = {
    "glioma": "A tumor arising from glial (supporting) brain tissue.",
    "meningioma": "A tumor arising from the meninges, the brain's outer membranes.",
    "pituitary": "A tumor of the pituitary gland, at the base of the brain.",
    "no_tumor": "No tumor detected in the scan.",
}
for column, class_name in zip(class_columns, config.CLASS_NAMES):
    with column:
        st.markdown(f"**{config.CLASS_DISPLAY_NAMES[class_name]}**")
        st.caption(class_descriptions[class_name])

st.divider()

# --------------------------------------------------------------------------
# How it works
# --------------------------------------------------------------------------

st.subheader("How it works")
how_col1, how_col2, how_col3 = st.columns(3)

with how_col1:
    st.markdown("**1. Model**")
    st.write(
        "EfficientNetB0, pretrained on ImageNet, with a custom "
        "classification head fine-tuned on labeled MRI images."
    )
with how_col2:
    st.markdown("**2. Evaluation**")
    st.write(
        "Measured on a held-out test set the model never trains or "
        "tunes on, with per-class precision, recall, and F1 - not just "
        "overall accuracy. Full numbers on the Results page."
    )
with how_col3:
    st.markdown("**3. Limitations**")
    st.write(
        "Trained and evaluated on one public dataset. Performance on "
        "scans from a different scanner or population is unknown. See "
        "the Methodology page for the full list of caveats."
    )

st.divider()

# --------------------------------------------------------------------------
# Get started
# --------------------------------------------------------------------------

st.subheader("Get started")
start_col1, start_col2, start_col3 = st.columns(3)
with start_col1:
    st.write("**Try it**")
    st.write("Upload a scan on the Classify page.")
with start_col2:
    st.write("**See the numbers**")
    st.write("Accuracy, per-class metrics, and training history on the Results page.")
with start_col3:
    st.write("**Read the methodology**")
    st.write("Data, preprocessing, architecture, and training details on the Methodology page.")

st.divider()

st.caption("Built with:")
logo_columns = st.columns(11)
logo_files = [
    "python.png",
    "keras.png",
    "tf.png",
    "opencv.png",
    "streamlit.png",
    "sklearn.png",
    "tqdm.png",
    "pandas.png",
    "seaborn.png",
    "numpy.png",
    "matplotlib.png",
]
for column, filename in zip(logo_columns, logo_files):
    column.image(Image.open(f"assets/image-logos/{filename}"))
