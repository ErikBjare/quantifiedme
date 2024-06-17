from datetime import datetime, timedelta
from functools import cache

import gradio as gr
import pandas as pd
from quantifiedme.derived.all_df import load_all_df
from quantifiedme.derived.sleep import load_sleep_df
from quantifiedme.load.qslang import load_daily_df as load_qslang_daily_df
from quantifiedme.load.qslang import load_df as load_qslang_df


@cache
def load_all(fast=True) -> pd.DataFrame:
    print(f"Loading all data {fast=}")
    df = load_all_df(fast)
    print("DONE! Loaded", len(df), "rows")
    return df


def load_timeperiods(tps: list[tuple[datetime, datetime]]) -> pd.DataFrame:
    fast = True
    now = datetime.now()
    oldest_start = min((start for start, _ in tps))
    if oldest_start < now - timedelta(days=30):
        fast = False
    df = load_all(fast=fast)
    print("Slicing timeperiods")
    df = pd.concat([df[(df.index >= start) & (df.index <= end)] for start, end in tps])
    print("DONE! Sliced", len(df), "rows")
    print(df)
    return df


def load_timeperiod(period: str) -> pd.DataFrame:
    return load_timeperiods([period_to_timeperiod(period)])


def period_to_timeperiod(period: str) -> tuple[datetime, datetime]:
    now = datetime.now()
    match period.lower():
        case "day":
            return (now.replace(hour=0, minute=0, second=0), now)
        case "week":
            return (
                now.replace(hour=0, minute=0, second=0) - timedelta(days=now.weekday()),
                now,
            )
        case "month":
            return (now.replace(day=1, hour=0, minute=0, second=0), now)
        case "year":
            return (now.replace(month=1, day=1, hour=0, minute=0, second=0), now)
        case _:
            raise ValueError(f"Unknown period: {period}")


def dropdown_dfcols(df) -> gr.Dropdown:
    # if no data, return empty choices
    if df.empty or len(df) == 0:
        return gr.Dropdown(choices=[], value=None)
    columns = [str(c) for c in df.columns if str(c) != "date"]
    return gr.Dropdown(choices=columns, value=columns[0])


def plot_cat(df: pd.DataFrame | None, col: str | None = None) -> gr.BarPlot:
    y_lim = None
    if col:
        assert df is not None
        print(f"Col changed to {col}")
        df = df.reset_index().rename(columns={"index": "date"})
        df = df[["date", col]]
        y_max = max([v for v in df[col] if isinstance(v, (int, float))], default=0)
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
            .drop(columns=["time:All_events", "time:All_cols"], errors="ignore")
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
            view_time()

        with gr.Tab("Sleep"):
            view_sleep()

        with gr.Tab("Drugs"):
            view_drugs()

        with gr.Tab("Correlations"):
            view_plot_correlations()

        with gr.Tab("Data sources"):
            view_sources()

    app.launch()


def _sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    # reorder column so that column with highest sum is first
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    cols = df.columns.tolist()
    cols.remove("date")
    cols = sorted(
        cols, key=lambda c: -df[c].sum() if df[c].dtype in [int, float] else 0
    )
    cols = ["date"] + cols
    return df[cols].set_index("date")


def _prepare_df_for_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().rename(columns={"index": "date"})
    print(df)
    print("Duplicates in date", df.duplicated("date").sum())
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.date
    df = df.sort_values("date", ascending=False)
    df = _sort_cols(df)
    return df


def dataframe_summary(range: str | None) -> gr.Dataframe:
    print(f"Loading summary for {range=}")
    df = load_timeperiod(range) if range else pd.DataFrame()
    df = _prepare_df_for_view(df)
    return gr.Dataframe(df)


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

    dataframe_el = dataframe_summary(None)
    btn.click(dataframe_summary, [range], dataframe_el)

    # when loaded
    # update the top categories plot
    dataframe_el.change(
        fn=plot_top_cats, inputs=[dataframe_el], outputs=[plot_top_cats_output]
    )


def dataframe_all(fast: bool) -> gr.Dataframe:
    print(f"Loading df_all {fast=}")
    df = load_all(fast=fast)
    df = _prepare_df_for_view(df)
    return gr.Dataframe(df)


def view_explore():
    """View to explore the raw data"""
    with gr.Group():
        gr.Markdown("Query options")
        fast = gr.Checkbox(value=True, label="Fast")
        btn = gr.Button(value="Load")

    dataframe_el = gr.Dataframe([])
    btn.click(dataframe_all, [fast], dataframe_el)

    with gr.Group():
        cols_dropdown = gr.Dropdown(
            label="Columns", choices=[], allow_custom_value=True, interactive=True
        )
        # when loaded, update the dropdown
        dataframe_el.change(
            fn=dropdown_dfcols, inputs=[dataframe_el], outputs=[cols_dropdown]
        )

        plot_cat_output = plot_cat(None, None)
        # when dropdown changes, update the plot
        cols_dropdown.change(
            fn=plot_cat, inputs=[dataframe_el, cols_dropdown], outputs=[plot_cat_output]
        )


def view_plot_correlations():
    """View to plot correlations between columns"""
    dataframe_el = gr.Dataframe(pd.DataFrame())

    with gr.Group():
        gr.Markdown("Query options")
        fast = gr.Checkbox(value=True, label="Fast")
        btn = gr.Button(value="Load")
        btn.click(dataframe_all, [fast], dataframe_el)

    def plot(df, col1: str | None, col2: str | None) -> gr.ScatterPlot:
        print(f"Plotting {df=} {col1=} {col2=}")
        if df is None or col1 is None or col2 is None:
            return gr.ScatterPlot(
                pd.DataFrame({"x": [], "y": []}), x="x", y="y", height=300
            )
        df = df[[col1, col2]].dropna()
        # df["corr"] = df[col1].corr(df[col2])
        return gr.ScatterPlot(
            df,
            x=col1.replace(":", "\\:"),
            y=col2.replace(":", "\\:"),
            height=300,
        )

    with gr.Row():
        cols1_dropdown = gr.Dropdown(
            label="X Column", choices=[], allow_custom_value=True, interactive=True
        )
        cols2_dropdown = gr.Dropdown(
            label="Y Column", choices=[], allow_custom_value=True, interactive=True
        )
        # when loaded, update the dropdown
        dataframe_el.change(
            fn=dropdown_dfcols,
            inputs=[dataframe_el],
            outputs=[cols1_dropdown],
        )
        dataframe_el.change(
            fn=dropdown_dfcols,
            inputs=[dataframe_el],
            outputs=[cols2_dropdown],
        )
    with gr.Tab("Plots"):
        plot_corr_output = plot(
            dataframe_el.value, cols1_dropdown.value, cols2_dropdown.value
        )
        btn = gr.Button(value="Run")
        btn.click(
            plot, [dataframe_el, cols1_dropdown, cols2_dropdown], plot_corr_output
        )


def view_drugs():
    """View to explore drugs data"""

    def load():
        daily_df = load_qslang_daily_df()
        df = load_qslang_df()
        return daily_df, df

    button_load = gr.Button("Load")

    daily_df, df = load()
    daily_df_el = gr.Dataframe(daily_df)
    df_el = gr.Dataframe(df)

    button_load.click(load, [], [daily_df_el, df_el])

    return daily_df_el, df_el


def view_sleep():
    """View to explore sleep data"""

    def load():
        df = load_sleep_df()
        return df

    button_load = gr.Button("Load")

    df = load()
    df_el = gr.Dataframe(df)

    button_load.click(load, [], [df_el])


def view_time():
    """View to explore time data"""
    pass


def view_sources():
    """View to show sources of data"""
    pass
