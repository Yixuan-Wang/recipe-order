from .data import DataBatchPerm
from utils.metrics import MetricsClassification

import torch

import shared.env as env
import shared.graph as graph
from dataclasses import dataclass, field


@dataclass
class MetricsPerm(MetricsClassification):
    loss: float = field(kw_only=True, default=torch.nan)

    def accept(
        self, *, prediction: torch.Tensor, target: torch.Tensor, batch: DataBatchPerm
    ):
        goal_prediction = prediction.argmax(-1)
        goal_target = target

        self.true += (goal_prediction == goal_target).sum().item()  # type: ignore
        self.total += goal_target.shape[-1]

        self.relevant += (relevant := goal_target != 0).sum().item()  # type: ignore
        self.positive += (positive := goal_prediction != 0).sum().item()  # type: ignore
        self.true_positive += (positive & relevant).sum().item()  # type: ignore

        loss = torch.nn.functional.cross_entropy(prediction, goal_target.to(dtype=torch.int64))
        self.loss = loss.item()
        return loss
