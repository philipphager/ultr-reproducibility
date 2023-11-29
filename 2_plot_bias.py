from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from flax import linen as nn
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from app.components import sidebar
from app.utils.file import parse_model_name, has_checkpoint
from app.utils.model import load_state
from app.utils.const import METRICS, MODELS, LOSSES


def get_synthetic_batch():
    k = st.sidebar.slider("Positions", 2, 25, 10)
    positions = (np.arange(k) + 1).reshape(1, -1)
    media_type = np.full((1, k), st.sidebar.slider("Media type", 1, 100, 1))
    displayed_time = np.full((1, k), st.sidebar.slider("Display time", 1, 17, 1))
    serp_height = np.full((1, k), st.sidebar.slider("SERP height", 1, 17, 1))
    slip_off = np.full((1, k), st.sidebar.slider("Slip off", 1, 11, 1))

    return {
        "position": positions.astype(int),
        "media_type": media_type.astype(int),
        "displayed_time": displayed_time.astype(int),
        "serp_height": serp_height.astype(int),
        "slipoff_count_after_click": slip_off.astype(int),
    }


def get_model(path: Path):
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path=str(path / ".hydra")):
        config = compose(config_name="config.yaml")
        return instantiate(config.model)


def get_examination(model, state, batch, normalize):
    examination_logits = model.apply(
        state["params"],
        batch,
        training=False,
        rngs={"dropout": state["dropout_key"]},
        method=model.predict_examination,
    )
    examination = nn.sigmoid(examination_logits)
    examination = examination / examination[0] if normalize else examination
    batch["examination"] = examination

    return pd.DataFrame(
        {
            "position": batch["position"][0],
            "media_type": batch["media_type"][0],
            "displayed_time": batch["displayed_time"][0],
            "serp_height": batch["serp_height"][0],
            "slipoff_count_after_click": batch["slipoff_count_after_click"][0],
            "examination": examination,
        }
    )


def plot_position_bias(df):
    models = st.sidebar.multiselect("Models", MODELS, default=df.model.unique())
    df = df[df.model.isin(models)]
    return (
        alt.Chart(df, width=800)
        .mark_line(point=True)
        .encode(
            x=alt.X("position:O").title("Position").axis(labelAngle=0),
            y=alt.Y("examination:Q").title("Examination").scale(domain=(0, 1)),
            color=alt.Color("name").title("Model"),
        )
        .configure_point(size=75)
    )


sidebar.draw()
st.markdown("# Inspect Bias")
st.info(
    "Sometimes Hydra fails to clear its context, you might need to refresh this page."
)

model_directory = Path(st.session_state["model_directory"])
directories = list(filter(has_checkpoint, map(Path, model_directory.glob("*/"))))
directories = [f for f in directories if "naive" not in f.name]

if len(directories) == 0:
    st.warning("No evaluation results to plot.")
    st.stop()

batch = get_synthetic_batch()
normalize = st.sidebar.toggle("Normalize bias by first position", False)

st.write(batch)
st.divider()

dfs = []

for path in directories:
    model = get_model(path)
    state = load_state(path)
    options = parse_model_name(path)

    df = get_examination(model, state, batch, normalize)
    df["model"] = options["model"]
    df["loss"] = options["loss"]
    df["name"] = f"{options['model']}, {options['loss']}"
    dfs.append(df)

if len(dfs) > 0:
    df = pd.concat(dfs)
    chart = plot_position_bias(df)
    st.write(chart)
