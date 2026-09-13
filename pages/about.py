"""About - project motivation and developer background."""

import streamlit as st

st.set_page_config(page_title="About - NeuroNet MRI", page_icon="👩‍💻", layout="wide")
st.title("About")

col1, col2 = st.columns(2)

col1.subheader("Mikayla Norton")
col1.text("Data scientist")
col1.markdown(
    "Hi! I'm Mikayla, the developer of NeuroNet MRI. I built this project "
    "as a second-year MS student in Data Science, and I've continued "
    "refactoring it since as a way to practice turning a strong student "
    "project into something closer to a well-engineered application."
)
col2.image("assets/headshot.jpg")

col1.markdown("#### Motivation")
col1.write(
    "My interest in this project comes from my own experience with "
    "hydrocephalus, a brain condition that gave me a personal stake in "
    "neuroimaging and a strong motivation to understand how machine "
    "learning could support - never replace - careful clinical review."
)
col1.write(
    "The goal isn't to claim this tool could diagnose anyone. It's to "
    "practice building the kind of evaluation, engineering, and honest "
    "reporting that responsible use of ML in a sensitive domain requires."
)
