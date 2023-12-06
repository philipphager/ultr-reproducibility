import altair as alt
import pandas as pd
import streamlit as st
import wandb

HYPERPARAMETERS = [
    "lr",
    "model/relevance_dims",
    "model/relevance_layers",
    "model/relevance_dropout",
    "model/bias_dims",
    "model/bias_layers",
    "model/bias_dropout",
]

METRICS = [
    "Val/click_loss",
    "Val/dcg@01",
    "Val/dcg@03",
    "Val/dcg@05",
    "Val/dcg@10",
    "Val/ndcg@10",
    "Val/mrr@10",
    "Test/dcg@01",
    "Test/dcg@03",
    "Test/dcg@05",
    "Test/dcg@10",
    "Test/ndcg@10",
    "Test/mrr@10",
    "Train/loss",
]


@st.cache_data()
def download_data(entity, project, run):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"config.config.run_name": run})
    return pd.concat([parse(run) for run in runs])


def get_available_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    configs = [r.config for r in runs if "config" in r.config]

    return sorted(set([c["config"]["run_name"] for c in configs]))


def parse(run):
    history_df = run.history(pandas=True)

    if len(history_df) > 0:
        metric_df = best_metric(history_df)
    else:
        st.warning("Run history not found, parsing metrics from an incomplete run.")
        metric_df = pd.json_normalize(run.summary._json_dict, sep="")

    config_df = pd.json_normalize(run.config, sep="/")
    return pd.concat([config_df, metric_df], axis=1)


def best_metric(history_df):
    columns = get_available_metrics(history_df)
    aggregation = {c: "min" if "loss" in c.lower() else "max" for c in columns}
    history_df = history_df.agg(aggregation).to_frame().T
    history_df.columns = [c.replace(".", "") for c in history_df.columns]
    return history_df


def get_available_metrics(df, stages=["train", "val", "test"]):
    filter_metrics = lambda x: any([i in x.lower() for i in stages])
    return list(filter(filter_metrics, df.columns))


def plot(df, x, y, group, plot_ci=True):
    base = alt.Chart(df, width=400)

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


run_names = get_available_runs("cm-offline-metrics", "baidu-reproducibility")
run_names = [r for r in run_names if "grid" in r]
selected_runs = st.sidebar.multiselect("Select W&B run:", run_names)

df = pd.concat(
    [
        download_data("cm-offline-metrics", "baidu-reproducibility", r)
        for r in selected_runs
    ]
)

val_metrics = get_available_metrics(df, stages=["val"])
val_metric = st.selectbox("Val metric", val_metrics)
test_metrics = get_available_metrics(df, stages=["test"])
minimize = st.checkbox("Minimize val", False)
test_metric = st.selectbox("Test metric", test_metrics)


df[HYPERPARAMETERS] = df[HYPERPARAMETERS].fillna(-1)
param_df = df.groupby(["model/_target_"] + HYPERPARAMETERS).agg(
    val_metric=(val_metric, "mean"),
).reset_index()

param_df = (
    param_df.sort_values(["model/_target_", "val_metric"], ascending=minimize)
    .groupby(["model/_target_"])
    .head(1)
)

columns = ["model/_target_"] + HYPERPARAMETERS

df = df.merge(param_df[columns], on=columns)

df["model/_target_"] = df["model/_target_"].map(lambda x: x.replace("src.models.", ""))
df["name"] = df["model/_target_"] + " - " + df["loss/_target_"]

base = alt.Chart(df, width=800, height=400)

chart = base.mark_bar().encode(
    x=alt.X("name").title("model"),
    y=alt.Y(f"mean({test_metric})").title(test_metric),
    color=alt.Color("loss/_target_").title("loss"),
) + base.mark_errorbar().encode(
    x=alt.X("name").title("model"),
    y=alt.Y(f"{test_metric}").title(test_metric),
    strokeWidth=alt.value(2)
)

st.write(chart)
