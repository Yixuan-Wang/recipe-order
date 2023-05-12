from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING, Literal, Optional, TypedDict
import numpy as np
from pyrsistent import v
import torch
import pandas as pd
import shared.env
import shared.graph as graph

from dataclasses import dataclass, field

from utils.stub import dc_copy_shallow, tokenize

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


@dataclass
class DataRaw:
    id: int = field(kw_only=True)
    id_label: str = field(kw_only=True)

    input_ids: torch.Tensor = field(kw_only=True)
    attention_mask: torch.Tensor = field(kw_only=True)
    target: Optional[torch.Tensor] = field(kw_only=True)

    # NOT_MASKED: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    def get_input(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }

    def move(self, device: torch.device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        if self.target is not None: 
            self.target = self.target.to(device)

    @staticmethod
    def collate_one(data):
        return data


@dataclass
class DataRawList:
    id: int = field(kw_only=True)
    id_label: str = field(kw_only=True)

    input_ids: list[torch.Tensor] = field(kw_only=True)
    attention_mask: list[torch.Tensor] = field(kw_only=True)
    target: Optional[torch.Tensor] = field(kw_only=True)

    view: slice = field(kw_only=True, default_factory=lambda: slice(None))
    """View on **permutation** of the list."""

    def with_view(self, view: slice):
        cp = dc_copy_shallow(self)
        cp.view = view
        return cp

    def __len__(self):
        return len(self.input_ids)


class DatasetMmresOption(TypedDict):
    max_graph_size: int
    min_graph_size: int


class DatasetMmres:
    def __init__(self, option: DatasetMmresOption, tokenizer: PreTrainedTokenizerFast):
        self.option = option
        # self.rng = np.random.default_rng(option["seed"] if "seed" in option else None)
        self.tokenizer = tokenizer
        self.df = self.init_dataframe()

    def init_dataframe(self):
        df = pd.read_json(shared.env.PATH_DATA_MMRES)

        # if not shared.env.INSTRUCTION_IS_ORDERED:
        #     df["shuffle_instr"] = df.apply(self.shuffle_graph_order, axis=1) # type: ignore

        df["graph_size"] = df["instrs"].apply(len)
        df = df[
            (df["graph_size"] <= self.option["max_graph_size"])
            & (df["graph_size"] >= self.option["min_graph_size"])
            & (df["edges"].apply(len) != 0)
        ].reset_index(drop=True)

        df["adj"] = pd.Series(
            graph.get_adj_matrix_from_edges(
                edges=edges,
                graph_size=graph_size,
            )
            for edges, graph_size in zip(df["edges"], df["graph_size"])
        )

        def tokenize(input: list[str]):
            result = self.tokenizer(input, return_token_type_ids=False)
            return {
                k: list(map(torch.tensor, v)) if isinstance(v, list) else v
                for k, v in result.items()
            }

        df = df.join(pd.DataFrame.from_records(df["instrs"].apply(tokenize)))

        return df
