from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypedDict
import torch


class MetricsClassificationResult(TypedDict):
    precision: float
    recall: float
    accuracy: float
    f1: float


@dataclass
class MetricsClassification:
    total: int = field(kw_only=True, default=0)
    """Number of total elements."""

    true: int = field(kw_only=True, default=0)
    """Number of matched elements."""

    positive: int = field(kw_only=True, default=0)
    """Number of predicted concerned elements."""

    true_positive: int = field(kw_only=True, default=0)
    """Number of predicted concerned elements that matches."""

    relevant: int = field(kw_only=True, default=0)
    """Number of real concerned elements."""

    def __add__(self, rhs) -> MetricsClassification:
        if rhs is None:
            return self
        elif not isinstance(rhs, MetricsClassification):
            raise ValueError(rhs)
        else:
            return MetricsClassification(
                total=self.total + rhs.total,
                true=self.true + rhs.true,
                true_positive=self.true_positive + rhs.true_positive,
                positive=self.positive + rhs.positive,
                relevant=self.relevant + rhs.relevant,
            )

    def accuracy(self):
        return float(torch.div(self.true, self.total))

    def precision(self):
        return float(torch.div(self.true_positive, self.positive))

    def recall(self):
        return float(torch.div(self.true_positive, self.relevant))

    def calc(self) -> MetricsClassificationResult:
        p = self.precision()
        r = self.recall()
        f1 = float(torch.div(2 * p * r, p + r))

        return {
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": self.accuracy(),
        }
