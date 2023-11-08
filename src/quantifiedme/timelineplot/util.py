from typing import TypeVar
from collections.abc import Iterable, Generator

T = TypeVar("T")


def take_until_next(
    ls: Iterable[T],
) -> Generator[tuple[tuple[int, int], T], None, None]:
    """
    Given an iterable with duplicate entries, chunk them together and return
    each chunk with its start and stop index.
    """
    last_v = None
    last_i = 0
    i = None
    v = None
    for i, v in enumerate(ls):
        if v == last_v:
            continue
        elif last_v is not None:
            yield (last_i, i - 1), last_v
        last_v = v
        last_i = i
    if v and i is not None and i != last_i:
        yield (last_i, i), v


def test_take_until_next():
    ls = [1, 1, 1, 2, 3, 3]
    assert [((0, 2), 1), ((3, 3), 2), ((4, 5), 3)] == list(take_until_next(ls))
