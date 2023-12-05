import altair as alt
import pandas as pd
import streamlit as st
import wandb

st.set_page_config(layout="wide")

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


def get_available_metrics(df):
    filter_metrics = lambda x: any([i in x.lower() for i in ["train", "val", "test"]])
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
selected_run_name = st.sidebar.selectbox("Select W&B run:", run_names)

df = download_data("cm-offline-metrics", "baidu-reproducibility", selected_run_name)
metrics = get_available_metrics(df)

meta_df = df.groupby(["model/_target_", "loss/_target_"]).agg(
    random_states=("random_state", "nunique"),
    runs=("random_state", "count"),
).reset_index()

st.write(meta_df)

x = st.selectbox("Hyperparameter:", HYPERPARAMETERS)
selected_metrics = st.multiselect("Metric:", METRICS, default=METRICS[0])
group = st.selectbox("Group by:", ["model/_target_", "random_state"] + HYPERPARAMETERS)
plot_ci = st.checkbox("Plot CI:", value=True)

st.divider()

row = []

for y in selected_metrics:
    row.append(plot(df, x, y, group, plot_ci))

chart = alt.hconcat(*row)
st.write(chart)
