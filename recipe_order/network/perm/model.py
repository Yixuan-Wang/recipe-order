from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding, PreTrainedModel, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

if TYPE_CHECKING:
    from .data import DataBatchPerm

import shared.constant as constant
import shared.env as env

logging.set_verbosity_error()


class ModelPerm(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert: PreTrainedModel = AutoModel.from_pretrained(
            env.PRETRAINED_MODEL_PERM
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 2),
            nn.Sigmoid(),
        )

        for param in self.bert.encoder.layer[:9].parameters():  # type: ignore
            param.requires_grad_(False)
        for param in self.bert.embeddings.parameters():  # type: ignore
            param.requires_grad_(False)

    def forward(self, input: DataBatchPerm):
        bert_output: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            **input.get_input()
        )

        prediction = self.classifier(bert_output.pooler_output)

        return prediction
