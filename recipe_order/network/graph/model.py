from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch
import torch.nn as nn
from typing import cast
from collections.abc import Iterable, Sequence
import dgl
from dgl.nn.pytorch import GINConv
import itertools
import functools
import einops
from einops.layers.torch import Rearrange

import shared.env as env
import shared.constant as constant

MODEL_EDGE_WEIGHT = "edge_weight"
MODEL_NODE_EMBEDDING = "node_embed"


# class ModelConnector():
#     def __init__(self) -> None:
#         super().__init__()
#         # self.tokenizer = AutoTokenizer.from_pretrained(env.PRETRAINED_MODEL_GRAPH_ISOMORPHISM, fast=True)
#         self.bert = AutoModel.from_pretrained(env.PRETRAINED_MODEL_GRAPH_ISOMORPHISM)
#         self.model_relative_ordering = train.relative.model.ModelRelativeOrdering()

#     def pooling(self, input: BatchEncoding):
#         with torch.no_grad():
#             pooling = cast(
#                 BaseModelOutputWithPoolingAndCrossAttentions,
#                 self.bert(
#                     **{
#                         key: val
#                         for key, val in input
#                         if key in constant.BERT_MODEL_INPUT_KEYS
#                     }
#                 ),
#             ).pooler_output

#         return pooling

#     def graph(self, pooling: torch.FloatTensor):
#         sentence_count = pooling.size(0)


@dataclass
class ModelOptionGraphIsomorphism:
    bert_hidden_size: int = field(kw_only=True, default=768)
    """Hidden state size of the last layer of the BERT model."""

    gin_layer: Sequence[int] = field(kw_only=True)
    """Layer configuration of each GIN. The $k$-th element of the list is the layer size of the $k$-th GIN."""

    gin_hidden_size: int = field(kw_only=True)
    """Hidden size of each layer of the GIN."""

    gin_epsilon: int = field(kw_only=True)
    """Hyperparameter $\\epsilon$, weights of the center node in aggregation"""


class ModelGraphIsomorphismOne(nn.Module):
    def __init__(self, option: ModelOptionGraphIsomorphism, k: int) -> None:
        super().__init__()

        layer_count = option.gin_layer[k]

        self.layers = nn.ModuleList(
            GINConv(
                nn.Linear(
                    in_features=option.bert_hidden_size
                    if layer == 0
                    else option.gin_hidden_size,
                    out_features=option.gin_hidden_size,
                ),
                aggregator_type="sum",
                init_eps=option.gin_epsilon,
                activation=nn.functional.relu,
            )
            for layer in range(layer_count)
        )
        self.readout = nn.Sequential(
            Rearrange("sentence layer repr -> sentence (layer repr)"),
            nn.ReLU(),
            nn.Linear(
                in_features=layer_count * option.gin_hidden_size,
                out_features=option.gin_hidden_size,
            ),
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        #: $[h^1, h^2, \dots]$
        layers: list[torch.Tensor] = functools.reduce(
            lambda hs, layer: (
                hs.append(layer(graph, hs[-1], graph.edata[MODEL_EDGE_WEIGHT])),  # type: ignore
                hs,
            )[-1],
            self.layers,
            [cast(torch.Tensor, graph.ndata[MODEL_NODE_EMBEDDING])],
        )

        #: $[h^1 \mathop{;} h^2 \mathop{;} \dots]$
        readout: torch.Tensor = self.readout(layers)

        return readout


class ModelGraphIsomorphism(nn.Module):
    def __init__(self, option: ModelOptionGraphIsomorphism) -> None:
        super().__init__()

        gin_count = len(option.gin_layer)

        self.gins = nn.ModuleList(
            ModelGraphIsomorphismOne(option, k) for k in range(gin_count)
        )
        self.fuse = nn.Sequential(
            Rearrange("setence k readout -> sentence (k readout)"),
            nn.Linear(
                in_features=gin_count * option.gin_hidden_size,
                out_features=1,
            ),
        )

    def forward(self, graphs: Iterable[dgl.DGLGraph]):
        # einops.pack()
        graph_readouts = [network(graph) for network, graph in zip(self.gins, graphs)]

        score: torch.Tensor = self.fuse(graph_readouts)
        prediction = score.argsort(descending=True)  # sentence -> sentence

        # this is not ordering?
