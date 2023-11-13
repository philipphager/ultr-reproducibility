import numpy as np
import pandas as pd
import streamlit as st
from flax import linen as nn

from app.components import sidebar
from app.utils.file import get_model_directories, parse_model_name
from app.utils.model import load_model, load_state


def get_position_bias(model, state, k: int, normalize: bool):
    positions = np.arange(1, k + 1)
    batch = {"position": positions}

    examination_logits = model.apply(
        state["params"],
        batch,
        training=False,
        rngs={"dropout": state["dropout_key"]},
        method=model.predict_examination,
    )
    examination = nn.sigmoid(examination_logits)
    examination = examination / examination[0] if normalize else examination
    return pd.DataFrame(
        {
            "position": positions,
            "examination": examination,
        }
    )


sidebar.draw()
st.markdown("# PBM Bias")

model_directory = st.session_state["model_directory"]
model_directories = get_model_directories(model_directory)

if len(model_directories) == 0:
    st.warning("No evaluation results to plot.")
    st.stop()

k = st.slider("Positions", 1, 25, 10)
normalize = st.toggle("Normalize bias")

dfs = []

for path in model_directories:
    model = load_model(path)
    state = load_state(path)
    options = parse_model_name(path)

    df = get_position_bias(model, state, k, normalize)
    df["model"] = options["name"]
    df["loss"] = options["loss"]
    df["name"] = f"{options['model']}, {options['loss']}"
    dfs.append(df)

with st.expander("Inspect raw data"):
    top_n = st.number_input("Top n", 1_000, step=100)
    st.write(df.head(top_n))
