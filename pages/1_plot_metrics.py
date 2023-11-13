import glob
from pathlib import Path
from typing import Union

import altair as alt
import pandas as pd
import streamlit as st

from pages.components import sidebar

sidebar.draw()

st.markdown("# Test Metrics")

MODELS = ["pbm", "two-towers", "naive"]
METRICS = ["dcg@01", "dcg@03", "dcg@05", "dcg@10", "ndcg@10", "mrr@10"]
LOSSES = ["listwise", "listwise-dla", "pointwise", "pointwise-em"]


def is_complete(path: Union[str, Path]) -> bool:
    path = Path(path)
    state_path = path / "best_state"
    config_path = path / ".hydra/config.yaml"
    return config_path.exists() and state_path.exists()


def parse_name(path: Path):
    directory = path.name
    options = {}

    for option in directory.split(","):
        k, v = option.split("=")
        options[k] = v

    return options


def get_test_results(path: Path) -> pd.DataFrame:
    model_directories = glob.glob(str(path / "*/"))
    model_directories = list(filter(is_complete, model_directories))

    dfs = []

    if len(model_directories) == 0:
        st.warning("No test.parquet files available")
        st.stop()

    for directory in model_directories:
        directory = Path(directory)
        options = parse_name(directory)

        df = pd.read_parquet(directory / "test.parquet")
        df["model"] = options["model"]
        df["loss"] = options["loss"]
        df["name"] = f"{options['model']}, {options['loss']}"
        dfs.append(df)

    return pd.concat(dfs)


def plot_metrics(df):
    metric = st.selectbox("Metric", METRICS)
    models = st.multiselect("Models", MODELS)
    df = df[df.model.isin(models)]

    color_by = st.selectbox("Color by", ["model", "loss"])
    color_domain = MODELS if color_by == "model" else LOSSES

    # Find global sort order
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
test_df = get_test_results(model_directory)

with st.expander("Inspect raw data"):
    top_n = st.number_input("Top n", 1_000, step=100)
    st.write(test_df.head(top_n))

chart = plot_metrics(test_df)
st.write(chart)
