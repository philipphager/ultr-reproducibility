from pathlib import Path

import streamlit as st


def assert_directory_exists(path: Path):
    if not path.exists():
        st.sidebar.error(f"Could not find: {path}")
        st.stop()


def get_model_directory() -> Path:
    model_directory = st.sidebar.text_input("Model directory", "multirun/")
    return Path(model_directory)


def draw():
    # Set global variables into session state
    model_directory = get_model_directory()
    assert_directory_exists(model_directory)
    
    st.session_state["model_directory"] = model_directory
