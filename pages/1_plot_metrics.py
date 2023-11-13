from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from pages.components import sidebar
from pages.utils.const import METRICS, MODELS, LOSSES
from pages.utils.file import get_model_directories, parse_model_name

sidebar.draw()

st.markdown("# Test Metrics")


def get_test_results(paths: List[Path]) -> pd.DataFrame:
    dfs = []

    for path in paths:
        options = parse_model_name(path)

        df = pd.read_parquet(path / "test.parquet")
        df["model"] = options["model"]
        df["loss"] = options["loss"]
        df["name"] = f"{options['model']}, {options['loss']}"
        dfs.append(df)

    return pd.concat(dfs)


def plot_metrics(df):
    metric = st.selectbox("Metric", METRICS)
    models = st.multiselect("Models", MODELS, default=df.model.unique())
    df = df[df.model.isin(models)]

    color_by = st.selectbox("Color by", ["model", "loss"])
    color_domain = MODELS if color_by == "model" else LOSSES

    # Find global sorting order
    top_df = df.groupby("name")[metric].mean().reset_index()
    top_df = top_df.sort_values([metric], ascending=False)
    sort = list(top_df.name)

    bars = (
        alt.Chart(df, width=700, height=400)
        .mark_bar()
        .encode(
            x=alt.X("name:N").axis(labelAngle=-45, labelOverlap=False).sort(sort),
            y=alt.Y(f"mean({metric})").title(metric),
            color=alt.Color(color_by).title(color_by).scale(domain=color_domain),
        )
    )

    error = (
        alt.Chart(df, width=700, height=400)
        .mark_errorbar(extent="ci")
        .encode(
            x=alt.X("name:N").sort(sort),
            y=alt.Y(f"{metric}").title(metric),
            strokeWidth=alt.value(2),
        )
    )

    return bars + error


model_directory = st.session_state["model_directory"]
model_directories = get_model_directories(model_directory)

if len(model_directories) == 0:
    st.warning("No evaluation results to plot.")
    st.stop()

test_df = get_test_results(model_directories)

with st.expander("Inspect raw data"):
    top_n = st.number_input("Top n", 1_000, step=100)
    st.write(test_df.head(top_n))

chart = plot_metrics(test_df)
st.write(chart)
