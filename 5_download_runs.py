from io import BytesIO

import pandas as pd
import streamlit as st
import wandb


def get_available_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    configs = [r.config for r in runs if "config" in r.config]
    return sorted(set([c["config"]["run_name"] for c in configs]))


def cross_join(df1, df2):
    df1["key"] = 0
    df2["key"] = 0
    df = df1.merge(df2, on="key", how="outer")
    return df.drop(columns=["key"])


def download_run(entity, project, run_name):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"config.config.run_name": run_name})
    dfs = []

    bar = st.progress(0, text=f"Downloaded: {run_name}")

    for i, run in enumerate(runs):
        history_df = run.history(pandas=True)

        if len(history_df) == 0:
            st.warning("Skipped an incomplete run")
            continue

        # Ignore test columns, as test metrics were only computed based on DCG@10
        # and might be misleading:
        columns = [c for c in history_df.columns if "test" not in c.lower()]
        history_df = history_df[columns]

        # Rename columns:
        history_df.columns = history_df.columns.str.replace(".", "")
        history_df = history_df.rename(columns={"_step": "Misc/Epoch"})

        config_df = pd.json_normalize(run.config, sep="/")
        df = cross_join(config_df, history_df)
        df["run"] = run_name
        dfs.append(df)

        bar.progress(
            (i + 1) / len(runs), text=f"Downloaded: {run_name} {(i + 1)}/{len(runs)}"
        )

    return pd.concat(dfs)


ENTITY = "cm-offline-metrics"
PROJECT = "baidu-reproducibility"

runs = get_available_runs(ENTITY, PROJECT)
default_runs = [r for r in runs if "Baidu" in r]
selected_runs = st.multiselect("Select W&B run:", runs, default=default_runs)
df = None

if "data" in st.session_state:
    df = st.session_state["data"]
else:
    if st.button("Start download"):
        df = pd.concat([download_run(ENTITY, PROJECT, run) for run in selected_runs])
        st.session_state["data"] = df

if df is not None:
    st.table(df.head())
    f = BytesIO()
    df.to_parquet(f)
    st.download_button("Download", f, file_name="wandb.parquet")
