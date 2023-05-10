from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TypeVar
from dataclasses import fields

_T = TypeVar("_T")


def pad_to(
    seq: Sequence,
    to: int,
    value: float,
) -> torch.Tensor:
    t = torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq
    return torch.nn.functional.pad(t, (0, to - len(seq)), value=value)


def dc_asdict_shallow(obj):
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def dc_copy_shallow(obj: _T) -> _T:
    return type(obj)(**dc_asdict_shallow(obj))
