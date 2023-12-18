import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")


BASE_MODELS = [
    "baidu-ultr_baidu-base-12L",
    "baidu-ultr_tencent-bert-12L",
]

HYPERPARAMETERS = [
    "lr",
    "model/relevance_dims",
    "model/relevance_layers",
    "model/relevance_dropout",
    "model/bias_dims",
    "model/bias_layers",
    "model/bias_dropout",
]

LOSSES = [
    "pointwise",
    "listwise",
    "pointwise-em",
    "listwise-dla",
    "pointwise-ipw",
    "pairwise-ipw",
    "listwise-ipw",
]

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

DEFAULT_METRICS = [
    "Val/click_loss",
    "Val/ndcg@10",
    "Val/BC_ndcg@10",
]

MODELS = [
    "NaiveModel",
    "PositionBasedModel",
    "TwoTowerModel",
]


@st.cache_data()
def load_best_val_metric_per_run(path: str, metric: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    minimize = "loss" in metric.lower()

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


@st.cache_data()
def filter_runs(
    df: pd.DataFrame,
    base_model: str,
    ignore_early_stopping: bool,
):
    df = df[df["base_model"] == base_model]

    if ignore_early_stopping:
        df = df[df.max_epochs <= df.es_patience]

    return df


@st.cache_data()
def preprocess_runs(
    df: pd.DataFrame,
):
    loss2name = {
        "rax.softmax_loss": "listwise",
        "rax.pointwise_sigmoid_loss": "pointwise",
        "src.loss.regression_em": "em",
        "src.loss.dual_learning_algorithm": "dla",
        "src.loss.inverse_propensity_weighting": "ipw",
        "pointwise_mse_loss": "mse",
    }

    df["base_model"] = df["data/name"].str.split("/").map(lambda x: x[-1])
    df["model"] = df["model/_target_"].str.split(".").map(lambda x: x[-1])

    loss_prefix = df["loss/loss_fn/_target_"].map(loss2name) + "-"
    df["loss"] = loss_prefix.fillna("") + df["loss/_target_"].map(loss2name)

    return df


@st.cache_data()
def plot(df, x, y, group, plot_ci=True):
    base = alt.Chart(df, width=600, height=400)

    chart = base.mark_line(point=True, radius=10).encode(
        x=alt.X(f"{x}:O"),
        y=alt.Y(f"mean({y}):Q").scale(zero=False),
        color=f"{group}:N" if group != "None" else None,
    )

    if plot_ci:
        chart += base.mark_errorband(extent="ci").encode(
            x=alt.X(f"{x}:O"),
            y=alt.Y(f"{y}:Q").scale(zero=False),
            color=f"{group}:N" if group != "None" else None,
        )

    return chart


st.sidebar.markdown("### Load hyperparameters:")
file = st.sidebar.text_input("File:", value="multirun/hyperparameters.parquet")

st.sidebar.markdown("### Filter runs:")
base_model = st.sidebar.selectbox("Base model", BASE_MODELS)
metric = st.sidebar.selectbox("Select best run based on:", METRICS)
models = st.sidebar.multiselect("Model", MODELS, default=["NaiveModel"])
losses = st.sidebar.multiselect("Loss", LOSSES, default=["pointwise"])
ignore_early_stopping = st.sidebar.toggle("Ignore runs with early stopping")

st.markdown("### Plot hyperparameters:")
parameter = st.selectbox("Hyperparameter", HYPERPARAMETERS)
group = st.selectbox("Group by", ["model", "loss", "random_state"] + HYPERPARAMETERS)
metrics = st.multiselect("Show metrics:", METRICS, default=DEFAULT_METRICS)
plot_ci = st.toggle("Bootstrap confidence interval:", True)
st.divider()

df = load_best_val_metric_per_run(file, metric)
df = preprocess_runs(df)
df = filter_runs(df, base_model=base_model, ignore_early_stopping=ignore_early_stopping)

source = df[(df.model.isin(models)) & (df.loss.isin(losses))]

st.sidebar.divider()
st.sidebar.markdown(f"""
**Runs:** {len(source)}\n
**Learning rates**: {source["lr"].unique()}\n
**Random states**: {source["random_state"].unique()}\n
""")

row = []

for y in metrics:
    chart = plot(source, x=parameter, y=y, group=group, plot_ci=plot_ci)
    row.append(chart)

chart = alt.hconcat(*row)
st.write(chart)
