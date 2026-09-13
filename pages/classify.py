"""Classify - upload an MRI scan and get a model prediction.

This page is intentionally thin: it validates the upload, calls
`src.inference.predict`, and renders the structured result. All ML
logic lives in `src/`, so this file has no TensorFlow imports or
prediction logic of its own (Phase 3.2).
"""

import streamlit as st

from src import config
from src.inference import predict
from src.model import get_cached_model
from src.validation import validate_upload

st.set_page_config(page_title="Classify - NeuroNet MRI", page_icon="🧠", layout="wide")
st.title("Classify")

st.info(
    "Model predictions are a research demonstration, not a medical "
    "diagnosis. Do not use this tool to make real health decisions.",
    icon="ℹ️",
)

uploaded_file = st.file_uploader(
    "Upload an MRI scan",
    type=["jpg", "jpeg", "png"],
    help=f"Max size {config.MAX_UPLOAD_SIZE_MB} MB. JPG, JPEG, or PNG.",
)
sample_images_url = "https://drive.google.com/drive/folders/1hBKM8E6SPLeYVhU7wuAIWDhhfd5EiEAn?usp=sharing"
st.caption(
    f"Don't have a scan to try? [Sample images]({sample_images_url}) are "
    f"available for exploration (not part of the training or test set)."
)

if uploaded_file is not None:
    validation_result = validate_upload(uploaded_file, uploaded_file.name)

    if not validation_result.ok:
        st.error(validation_result.error_message, icon="🚨")
    else:
        image = validation_result.image
        preview_col, result_col = st.columns(2)

        with preview_col:
            st.subheader("Uploaded image")
            st.image(image, width=450)

        with result_col:
            st.subheader("Result")
            with st.spinner("Running inference..."):
                model = get_cached_model()
                result = predict(image, model)

            if result["is_uncertain"]:
                st.warning(
                    "**Uncertain / needs review** - the model's top "
                    "prediction had lower confidence than usual. Treat "
                    "this result with extra caution.",
                    icon="⚠️",
                )

            st.markdown(f"#### Predicted class: {result['display_name']}")
            st.metric("Model confidence (top class)", f"{result['confidence']:.1%}")
            st.caption(
                "This is the model's raw softmax probability, not a "
                "calibrated or clinical measure of correctness. See the "
                "Methodology page for how this threshold was chosen."
            )

            st.write("Full probability breakdown:")
            probability_rows = {
                config.CLASS_DISPLAY_NAMES[name]: value
                for name, value in result["probabilities"].items()
            }
            st.bar_chart(probability_rows)
