import streamlit as st
from pages.components import sidebar

sidebar.draw()

st.markdown(
    """
# Inspect position bias
Plot examination per position.
"""
)

st.write(st.session_state["model_directory"])

