"""Model loading, centralized (Phase 4).

Kept in its own module so:
  * the model path and load logic live in exactly one place, and
  * Streamlit's ``@st.cache_resource`` wraps a thin function here rather
    than being scattered across pages, which also makes this importable
    from `training/evaluate.py` (no Streamlit dependency required there).
"""

from __future__ import annotations

from pathlib import Path

from src import config


def load_model(model_path: Path = config.MODEL_PATH):
    """Load the Keras model from disk.

    This is intentionally a plain function with no caching decorator -
    the Streamlit app wraps it with ``st.cache_resource`` (see
    ``get_cached_model`` below) so it loads once per server process, and
    the training/evaluation scripts can call it directly without pulling
    in Streamlit at all.
    """

    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. Run `training/train.py` to "
            f"produce one, or point config.MODEL_PATH at an existing file "
            f"(e.g. config.LEGACY_MODEL_PATH for the frozen baseline)."
        )
    return tf.keras.models.load_model(model_path)


def get_cached_model(model_path: Path = config.MODEL_PATH):
    """Streamlit-cached model accessor - the one the app pages should use.

    Using ``st.cache_resource`` ensures the (potentially large) model is
    loaded into memory once per app process rather than on every
    prediction (Phase 4 - previously the app reloaded the model on every
    single classify request).
    """

    import streamlit as st

    @st.cache_resource(show_spinner="Loading model...")
    def _load(path_str: str):
        return load_model(Path(path_str))

    return _load(str(model_path))
