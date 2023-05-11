from __future__ import annotations

from transformers import AutoTokenizer

from dataclasses import dataclass, field
import functools
from typing import TYPE_CHECKING, Optional
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

import shared.env
import shared.graph as graph
from shared.traindata import DataRaw, DatasetMmres
from shared.traindata import DatasetMmresOption
from utils.abc import MaskedMapping
from utils.stub import nonnull, pad_to
from utils.metrics import MetricsClassificationResult

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


@dataclass
class DataBatchPointer:
    original_size: list[torch.Size] = field(kw_only=True)
    batch_size: int = field(kw_only=True)
    max_size: tuple[int, int] = field(kw_only=True)

    id: list[int] = field(kw_only=True)
    id_label: list[str] = field(kw_only=True)

    input_ids: torch.Tensor = field(kw_only=True)
    attention_mask: torch.Tensor = field(kw_only=True)
    target: Optional[torch.Tensor] = field(kw_only=True)

    def get_input(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def move(self, device: torch.device, device_subsidary: torch.device):
        self.input_ids = self.input_ids.to(device_subsidary)
        self.attention_mask = self.attention_mask.to(device_subsidary)
        if self.target is not None:
            self.target = self.target.to(device)

    @staticmethod
    def collate(data: list[DataRaw]) -> DataBatchPointer:
        max_size_step = max(datum.input_ids.shape[0] for datum in data)
        max_size_token = max(datum.input_ids.shape[1] for datum in data)

        id = [datum.id for datum in data]
        id_label = [datum.id_label for datum in data]

        def pad_sentence(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(
                t,
                (0, max_size_token - t.shape[1], 0, max_size_step - t.shape[0]),
                value=0.0,
            )

        def pad_target(t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(
                t,
                (0, max_size_step - t.shape[0], 0, max_size_step - t.shape[0]),
                value=0.0,
            )

        original_size = [datum.input_ids.shape for datum in data]

        input_ids = torch.stack([pad_sentence(datum.input_ids) for datum in data])
        attention_mask = torch.stack(
            [pad_sentence(datum.attention_mask) for datum in data]
        )

        if data[0].target is not None:
            target = torch.stack([pad_target(nonnull(datum.target)) for datum in data])
        else:
            target = None

        return DataBatchPointer(
            original_size=original_size,
            batch_size=len(data),
            max_size=(max_size_step, max_size_token),
            id=id,
            id_label=id_label,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target,
        )


class DatasetPointer(DatasetMmres, Dataset[DataRaw]):
    def __init__(
        self, option: DatasetMmresOption, tokenizer: PreTrainedTokenizerFast
    ) -> None:
        super().__init__(option, tokenizer)

    def __getitem__(self, index: int):
        line = self.df.loc[index]

        pad_size = max([len(sent) for sent in line["input_ids"]])

        input_ids = torch.stack(
            [pad_to(sent, to=pad_size, value=0) for sent in line["input_ids"]]
        )
        attention_mask = torch.stack(
            [pad_to(sent, to=pad_size, value=0) for sent in line["attention_mask"]]
        )

        datum = DataRaw(
            id=index,
            id_label=line["id"],
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=line["adj"],
        )

        return datum

    def __len__(self):
        return len(self.df)
