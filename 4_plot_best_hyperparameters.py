import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

METRICS = [
    "Val/click_loss",
    "Val/BC_dcg@01",
    "Val/BC_dcg@03",
    "Val/BC_dcg@05",
    "Val/BC_dcg@10",
    "Val/BC_mrr@10",
    "Val/BC_ndcg@10",
    "Val/dcg@01",
    "Val/dcg@03",
    "Val/dcg@05",
    "Val/dcg@10",
    "Val/mrr@10",
    "Val/ndcg@10",
]


@st.cache_data()
def load_best_val_metric_per_run(
    path: str, metric: str, minimize: bool
) -> pd.DataFrame:
    df = pd.read_parquet(path)

    columns = [
        "data/name",
        "lr",
        "loss/_target_",
        "model/_target_",
        "model/relevance_dims",
        "model/relevance_layers",
        "model/relevance_dropout",
        "model/bias_dims",
        "model/bias_layers",
        "model/bias_dropout",
        "random_state",
    ]

    # Some models do not use all hyperparameters we group by, make sure they are set:
    df[columns] = df[columns].fillna(-1)

    return (
        df.sort_values(columns + [metric], ascending=minimize).groupby(columns).head(1)
    )


def plot(df, color):
    df["name"] = df["model/_target_"] + " - " + df["loss/_target_"]

    base = alt.Chart(df, width=900, height=450)

    bar = base.mark_bar().encode(
        x=alt.X("name", sort="x"),
        y=f"mean({plot_metric})",
        color=color,
    )
    error = base.mark_errorbar().encode(
        x=alt.X("name", sort="x"),
        y=plot_metric,
        strokeWidth=alt.value(2),
    )

    return bar + error


st.markdown("### Best Parameters")
metric = st.selectbox("Select best run based on:", METRICS)
minimize = "loss" in metric.lower()

df = load_best_val_metric_per_run("multirun/hyperparameters.parquet", metric, minimize)

columns = [
    "data/name",
    "model/_target_",
    "loss/_target_",
    "lr",
    "model/relevance_dims",
    "model/relevance_layers",
    "model/relevance_dropout",
    "model/bias_dims",
    "model/bias_layers",
    "model/bias_dropout",
]

param_df = (
    df.groupby(columns)
    .agg(metric=(metric, "mean"), counts=("loss/_target_", "count"))
    .reset_index()
    .sort_values(
        by=["data/name", "model/_target_", "loss/_target_", "metric"],
        ascending=minimize,
    )
    .groupby(["data/name", "model/_target_", "loss/_target_"])
    .head(1)
)

st.table(param_df.sort_values(["data/name", "model/_target_", "loss/_target_"]))

plot_metric = st.selectbox("Plot metric:", METRICS)
color = st.selectbox("Color by:", ["model/_target_", "loss/_target_"])

df = df.merge(param_df, on=columns)
chart = plot(df, color)
st.write(chart)
