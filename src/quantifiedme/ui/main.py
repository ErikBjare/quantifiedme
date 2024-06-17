import threading
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
from quantifiedme.derived.all_df import load_all_df


@lru_cache
def load_all(fast=True):
    df = load_all_df(fast)
    return df


# Load all data in background
threading.Thread(target=load_all, args=(True,)).start()


def load_all_cols():
    df = load_all()
    return df.columns


def df_loaded(df):
    # if no data, return empty choices
    if df.empty or len(df) == 0:
        return gr.update(choices=[], value=None)
    return gr.update(choices=[str(c) for c in df.columns], value=df.columns[0])


def col_changed(df, col: str):
    assert len(col) < 30, f"Column name too long: {col}"
    print(f"Col changed to {col}")
    df = df.reset_index().rename(columns={"index": "date"})
    df = df[["date", col]]
    print(df[col].dtype)
    print(df[col].head())
    gr.update(data=df, y=col.replace(":", "\\:"), y_lim=[0, max(df[col])])
    return df


# Function to display data sources summary
def data_sources_summary():
    # Options
    fast = gr.Checkbox(value=True, label="Fast")
    btn = gr.Button(value="Load")

    cols_dropdown = gr.Dropdown(
        label="Columns", choices=[], allow_custom_value=True, interactive=True
    )

    dataframe_el = gr.Dataframe()
    btn.click(load_all, [fast], dataframe_el)

    # when loaded, update the dropdown
    dataframe_el.change(fn=df_loaded, inputs=[dataframe_el], outputs=[cols_dropdown])

    output = gr.BarPlot(
        x="date",
        y="time\\:Work",
        # tooltip=["x", "y"],
        y_lim=[0, 10],
        width=350,
        height=300,
    )

    # when dropdown changes, update the plot
    cols_dropdown.change(
        fn=col_changed, inputs=[dataframe_el, cols_dropdown], outputs=[output]
    )

    return "Summary of data sources goes here..."


def plot(v, a):
    g = 9.81
    theta = a / 180 * 3.14
    tmax = ((2 * v) * np.sin(theta)) / g
    timemat = tmax * np.linspace(0, 1, 40)

    x = (v * timemat) * np.cos(theta)
    y = ((v * timemat) * np.sin(theta)) - ((0.5 * g) * (timemat**2))
    df = pd.DataFrame({"x": x, "y": y})
    return df


def main():
    with gr.Blocks() as app:
        # Summary
        with gr.Tab("Summary"):
            gr.Markdown(data_sources_summary())

        # Trajectory
        with gr.Tab("Trajectory"):
            gr.Markdown("Just an example of plot with inputs")
            plot_trajectory()

        # Data overview
        with gr.Tab("Data Overview"):
            gr.Markdown("Data Overview goes here...")

    app.launch()


def plot_trajectory():
    with gr.Row():
        speed = gr.Slider(1, 30, 25, label="Speed")
        angle = gr.Slider(0, 90, 45, label="Angle")
    with gr.Tab("Plots"):
        output = gr.LinePlot(
            x="x",
            y="y",
            overlay_point=True,
            tooltip=["x", "y"],
            x_lim=[0, 100],
            y_lim=[0, 60],
            width=350,
            height=300,
        )
        btn = gr.Button(value="Run")
        btn.click(plot, [speed, angle], output)
