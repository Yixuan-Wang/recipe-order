from __future__ import annotations

import functools
import itertools
import operator
from typing import TypeVar, ParamSpec
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import field as _field

T = TypeVar("T")
R = TypeVar("R")
K = TypeVar("K")
V = TypeVar("V")
P = ParamSpec("P")

def call(obj: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """Backport of Python 3.11 `operator.call`"""
    return obj(*args, **kwargs)

def repeat_init(init: Callable[[], R], n: int) -> Iterator[R]:
    return map(call, itertools.repeat(init, n)) # type: ignore

def take(iter: Iterator[T], n: int) -> Iterator[T]:
    """Take `n` elements from iterator."""
    return map(operator.itemgetter(0), zip(iter, range(n)))

def regroup(mappings: Sequence[Mapping[K, Sequence[V]]]) -> Mapping[K, list[V]]:
    """Flatten a sequence of mappings to mappings of sequences."""

    if len(mappings) == 0: return {}

    keys = mappings[0].keys()
    result = {key: [] for key in keys}
    for item in mappings:
        for key, val in item.items():
            result[key].extend(val)
    return result
