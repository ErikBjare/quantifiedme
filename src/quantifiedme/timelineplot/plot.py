import sys
from typing import TypeVar, Union
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import matplotlib.pyplot as plt

from .util import take_until_next

T = TypeVar("T")
Color = Union[str | tuple[float, float, float]]
Index = TypeVar("Index", int, datetime)
Limits = tuple[Index, Index]
Event = tuple[Limits, Color, str]


@dataclass
class Bar:
    title: str
    events: list[Event]
    show_label: bool


class TimelineFigure:
    def __init__(self, title=None, **kwargs):
        self.fig = plt.figure(**kwargs)
        self.ax = plt.gca()
        if title:
            self.ax.set_title(title)
        self.bars: list[Bar] = []

    def plot(self):
        # We're assuming all bars share the same index type
        index_example = self.bars[0].events[0][0][0]
        if isinstance(index_example, int):
            limits: Limits = (sys.maxsize, 0)
        elif isinstance(index_example, datetime):
            limits = (
                datetime(2100, 1, 1, tzinfo=timezone.utc),
                datetime(1900, 1, 1, tzinfo=timezone.utc),
            )
        else:
            raise ValueError(f"Unknown index type: {type(index_example)}")

        # Check that type assumption is true
        assert all(
            [
                isinstance(event[0][0], type(index_example))
                and isinstance(event[0][1], type(index_example))
                for bar in self.bars
                for event in bar.events
            ]
        )

        for bar_idx, bar in enumerate(self.bars):
            for event in bar.events:
                (start, end), color, label = event
                length = end - start
                plt.barh(-bar_idx, length, left=start, color=color)
                limits = (min(limits[0], start), max(limits[1], end))
                plt.text(
                    start + length / 2,
                    -bar_idx,
                    label,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        tick_idxs = list(range(0, -len(self.bars), -1))
        self.ax.set_yticks(tick_idxs)
        self.ax.set_yticklabels([bar.title for bar in self.bars])
        self.ax.set_xlim(*limits)

        plt.show()

    def add_bar(self, events: list[Event], title: str, show_label: bool = False):
        self.bars.append(Bar(title, events, show_label))

    def add_chunked(
        self,
        ls: Iterable[T],
        cmap: Callable[[T], Color],
        title: str,
        show_label: bool = False,
    ):
        """Optimized version of add_bar that takes care of identical subsequent values"""
        bars = [
            ((i_start, i_end + 1), cmap(v), str(v) if show_label else "")
            for (i_start, i_end), v in take_until_next(ls)
        ]
        self.add_bar(bars, title, show_label)
