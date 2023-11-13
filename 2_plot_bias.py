from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from flax import linen as nn
from hydra import initialize, compose
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from app.components import sidebar
from app.utils.file import get_model_directories, parse_model_name
from app.utils.model import load_state
from src.data import LabelEncoder, Digitize, collate_fn


def get_train_data():
    train_dataset = load_dataset(
        "philipphager/baidu-ultr-606k", name="clicks", split="train[:1%]"
    )
    train_dataset.set_format("numpy")

    encode_media_type = LabelEncoder()
    encode_serp_height = Digitize(0, 1024, 16)
    encode_displayed_time = Digitize(0, 128, 16)
    encode_slipoff = Digitize(0, 10, 10)

    def encode_bias(batch):
        batch["media_type"] = encode_media_type(batch["media_type"])
        batch["displayed_time"] = encode_displayed_time(batch["displayed_time"])
        batch["serp_height"] = encode_serp_height(batch["serp_height"])
        batch["slipoff_count_after_click"] = encode_slipoff(
            batch["slipoff_count_after_click"]
        )
        return batch

    return train_dataset.map(encode_bias)


def get_synthetic_batch():
    k = st.sidebar.slider("Positions", 1, 25, 10)
    positions = (np.arange(k) + 1).reshape(1, -1)
    media_type = np.full((1, k), st.sidebar.slider("Media type", 1, 10, 1))
    displayed_time = np.full((1, k), st.sidebar.slider("Display time", 1, 10, 1))
    serp_height = np.full((1, k), st.sidebar.slider("SERP height", 1, 10, 1))
    slip_off = np.full((1, k), st.sidebar.slider("Slip off", 1, 10, 1))

    return {
        "position": positions.astype(int),
        "media_type": media_type.astype(int),
        "displayed_time": displayed_time.astype(int),
        "serp_height": serp_height.astype(int),
        "slipoff_count_after_click": slip_off.astype(int),
    }


def get_model(path: Path):
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
model_directory = st.session_state["model_directory"]
model_directories = get_model_directories(model_directory)
model_directories = [f for f in model_directories if "naive" not in f.name]

if len(model_directories) == 0:
    st.warning("No evaluation results to plot.")
    st.stop()

data_type = st.sidebar.selectbox("Select data", ["Synthetic", "Random Train Batch"])

if data_type == "Synthetic":
    batch = get_synthetic_batch()
elif data_type == "Random Train Batch":
    st.sidebar.markdown("Press `r` to fetch next batch")
    train_dataset = get_train_data()
    loader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))

normalize = st.sidebar.toggle("Normalize bias by first position", False)

st.write(batch)
st.divider()

dfs = []

for path in model_directories:
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
