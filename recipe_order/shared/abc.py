from __future__ import annotations
from typing import Protocol
from collections.abc import Sequence
import torch


class Inferable(Protocol):
    def infer(self, sentences: Sequence[str]) -> torch.Tensor:
        ...
