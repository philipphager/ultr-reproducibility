from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from app.components import sidebar
from app.utils.const import METRICS, MODELS, LOSSES
from app.utils.file import parse_model_name, has_metrics

sidebar.draw()


def get_results(paths: List[Path], eval_type: str) -> pd.DataFrame:
    dfs = []

    for path in paths:
        options = parse_model_name(path)

        df = pd.read_parquet(path / f"{eval_type}.parquet")
        df["model"] = options["model"]
        df["loss"] = options["loss"]
        df["name"] = f"{options['model']}, {options['loss']}"
        dfs.append(df)

    return pd.concat(dfs)


def plot_metrics(df):
    metric = st.sidebar.selectbox("Metric", METRICS)
    models = st.sidebar.multiselect("Models", MODELS, default=df.model.unique())
    df = df[df.model.isin(models)]

    color_by = st.sidebar.selectbox("Color by", ["model", "loss"])
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
directories = list(filter(has_metrics, map(Path, model_directory.glob("*/"))))
eval_type = st.sidebar.selectbox("Evaluation", ["val", "test"])

st.title(eval_type.capitalize() + " Metrics")

if len(directories) == 0:
    st.warning("No evaluation results to plot.")
    st.stop()

test_df = get_results(directories, eval_type)

with st.expander("Inspect raw data"):
    top_n = st.number_input("Top n", 1_000, step=100)
    st.write(test_df.head(top_n))

chart = plot_metrics(test_df)
st.write(chart)
