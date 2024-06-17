from datetime import datetime, timedelta
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
from quantifiedme.derived.all_df import load_all_df


@lru_cache
def load_all(fast=True) -> pd.DataFrame:
    print("Loading all data")
    df = load_all_df(fast)
    print("DONE! Loaded", len(df), "rows")
    return df


def load_timeperiods(tps: list[tuple[datetime, datetime]]) -> pd.DataFrame:
    df = load_all()
    print("Slicing timeperiods")
    df = pd.concat([df[(df.index >= start) & (df.index <= end)] for start, end in tps])
    print("DONE! Sliced", len(df), "rows")
    print(df)
    return df


def load_timeperiod(period: str) -> pd.DataFrame:
    return load_timeperiods([period_to_timeperiod(period)])


def period_to_timeperiod(period: str) -> tuple[datetime, datetime]:
    now = datetime.now()
    match period:
        case "Day":
            return (now.replace(hour=0, minute=0, second=0), now)
        case "Week":
            return (
                now.replace(hour=0, minute=0, second=0) - timedelta(days=now.weekday()),
                now,
            )
        case "Month":
            return (now.replace(day=1, hour=0, minute=0, second=0), now)
        case "Year":
            return (now.replace(month=1, day=1, hour=0, minute=0, second=0), now)
        case _:
            raise ValueError(f"Unknown period: {period}")


def load_all_cols() -> list[str]:
    df = load_all()
    return [str(c) for c in df.columns]


def dropdown_dfcols(df) -> gr.Dropdown:
    # if no data, return empty choices
    if df.empty or len(df) == 0:
        return gr.Dropdown(choices=[], value=None)
    return gr.Dropdown(choices=[str(c) for c in df.columns], value=df.columns[0])


def plot_cat(df: pd.DataFrame | None, col: str | None = None) -> gr.BarPlot:
    y_lim = None
    if col:
        assert df is not None
        print(f"Col changed to {col}")
        df = df.reset_index().rename(columns={"index": "date"})
        df = df[["date", col]]
        y_max = max(df[col])
        y_lim = [0, round(y_max) + 1 if isinstance(y_max, (int, float)) else 1]
        col = col.replace(":", "\\:")
    else:
        col = ""
    return gr.BarPlot(
        df,
        x="date",
        y=col,
        y_lim=y_lim,
        width=350,
        height=300,
    )


def filter_df_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # filter columns by column prefix (like "time:")
    return df[[c for c in df.columns if c.startswith(prefix)]]


def plot_top_cats(df: pd.DataFrame | None) -> gr.BarPlot:
    if df is not None:
        df = (
            filter_df_cols(df, "time:")
            .drop(columns=["time:All_events", "time:All_cols"])
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={"index": "category", 0: "hours"})
        )
    else:
        df = pd.DataFrame({"category": [], "hours": []})
    return gr.BarPlot(
        df,
        x="category",
        y="hours",
        vertical=False,
        width=350,
        height=300,
    )


def main():
    with gr.Blocks(title="QuantifiedMe (gradio)") as app:
        # Title
        gr.Markdown(
            """
# QuantifiedMe
This is a dashboard to explore quantified self data.
        """.strip()
        )

        # Summary (day/week/month/year)
        with gr.Tab("Summary"):
            view_summary()

        with gr.Tab("Explore"):
            view_explore()

        with gr.Tab("Time"):
            gr.Markdown("TODO")

        with gr.Tab("Sleep"):
            gr.Markdown("TODO")

        with gr.Tab("Drugs"):
            gr.Markdown("TODO")

        with gr.Tab("Correlations"):
            gr.Markdown("Just an example of plot with inputs")
            view_plot_correlations()

        with gr.Tab("Data sources"):
            gr.Markdown("TODO")

    app.launch()


def view_summary():
    """View to show summary of data for the current day/week/month/year"""
    with gr.Group():
        gr.Markdown("Query options")
        range = gr.Dropdown(
            label="Range", choices=["Day", "Week", "Month", "Year"], value="Month"
        )
        btn = gr.Button(value="Load")

    with gr.Group():
        gr.Markdown("Top categories")
        plot_top_cats_output = plot_top_cats(None)

    dataframe_el = gr.Dataframe(pd.DataFrame())
    btn.click(load_timeperiod, [range], dataframe_el)

    # when loaded
    # update the top categories plot
    dataframe_el.change(
        fn=plot_top_cats, inputs=[dataframe_el], outputs=[plot_top_cats_output]
    )


def view_explore():
    """View to explore the raw data"""
    with gr.Group():
        gr.Markdown("Query options")
        fast = gr.Checkbox(value=True, label="Fast")
        btn = gr.Button(value="Load")

    cols_dropdown = gr.Dropdown(
        label="Columns", choices=[], allow_custom_value=True, interactive=True
    )

    dataframe_el = gr.Dataframe([])
    btn.click(load_all, [fast], dataframe_el)

    # when loaded, update the dropdown
    dataframe_el.change(
        fn=dropdown_dfcols, inputs=[dataframe_el], outputs=[cols_dropdown]
    )

    plot_cat_output = gr.BarPlot(
        x="date",
        y="time\\:Work",
        # tooltip=["x", "y"],
        y_lim=[0, 10],
        width=350,
        height=300,
    )

    # when dropdown changes, update the plot
    cols_dropdown.change(
        fn=plot_cat, inputs=[dataframe_el, cols_dropdown], outputs=[plot_cat_output]
    )


def view_plot_correlations():
    """View to plot correlations between columns"""

    def plot(v, a):
        g = 9.81
        theta = a / 180 * 3.14
        tmax = ((2 * v) * np.sin(theta)) / g
        timemat = tmax * np.linspace(0, 1, 40)

        x = (v * timemat) * np.cos(theta)
        y = ((v * timemat) * np.sin(theta)) - ((0.5 * g) * (timemat**2))
        df = pd.DataFrame({"x": x, "y": y})
        return df

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
            height=300,
        )
        btn = gr.Button(value="Run")
        btn.click(plot, [speed, angle], output)
