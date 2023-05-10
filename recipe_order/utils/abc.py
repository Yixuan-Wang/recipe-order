from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

class MaskedMapping(Mapping):
    NOT_MASKED: ClassVar[list[str]] = []

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.NOT_MASKED)

    def __len__(self):
        return len(self.NOT_MASKED)
