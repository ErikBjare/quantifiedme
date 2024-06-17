import gradio as gr
import numpy as np
import pandas as pd
from quantifiedme.derived.all_df import load_all_df


def load_all(fast):
    df = load_all_df(fast)
    print(df)
    print(df.index)
    return df


# Function to display data sources summary
def data_sources_summary():
    fast = gr.Checkbox(label="Fast")
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
    dataframe_el = gr.Dataframe()
    btn = gr.Button(value="Load")
    btn.click(load_all, [fast], dataframe_el)
    # btn.click(load_all, [fast], output)
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
