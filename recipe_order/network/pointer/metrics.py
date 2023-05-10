from .data import DataBatchPointer
from utils.metrics import MetricsClassification


import torch

import shared.env
import shared.graph as graph
from dataclasses import dataclass, field


@dataclass
class MetricsPointer(MetricsClassification):
    loss: float = field(kw_only=True, default=torch.nan)

    def accept(self, *, prediction: torch.Tensor, target: torch.Tensor, batch: DataBatchPointer):
        device = target.device

        prediction_mask = torch.stack([
            torch.nn.functional.pad(
                ~torch.ones((size[0], size[0]), dtype=torch.bool).tril(),
                (0, batch.max_size[0] - size[0], 0, batch.max_size[0] - size[0]),
                value=False,
            )
            for size
            in batch.original_size
        ]).to(device)

        if shared.env.INSTRUCTION_IS_ORDERED:
            folded_target = target
            goal_target = graph.reachability_matrix(target)

            folded_prediction = graph.build_graph_from_prediction(0.5, prediction)
        else:
            folded_target = graph.fold_matrix_to_triu(target)
            goal_target = graph.fold_matrix_to_triu(graph.reachability_matrix(target))


            folded_prediction = graph.fold_matrix_to_triu(graph.build_graph_from_prediction(0.5, graph.unfold_matrix_from_triu(prediction)))

        ma_goal_prediction = prediction.masked_select(prediction_mask)
        ma_goal_target = goal_target.masked_select(prediction_mask)
        ma_folded_prediction = folded_prediction.masked_select(prediction_mask)
        ma_folded_target = folded_target.masked_select(prediction_mask)

        self.true += (ma_folded_prediction == ma_folded_target).sum().item() # type: ignore
        self.total += ma_folded_target.shape[-1]

        self.relevant += (relevant := (ma_folded_target == 1)).sum().item() # type: ignore
        self.positive += (positive := (ma_folded_prediction == 1)).sum().item() # type: ignore
        self.true_positive += (positive & relevant).sum().item() # type: ignore

        loss = torch.nn.functional.mse_loss(ma_goal_prediction, ma_goal_target)
        self.loss = loss.item()
        return loss
