from functools import reduce
from math import sqrt
from typing import Optional, Protocol, cast
from dataclasses import dataclass, field
from collections.abc import Callable
from einops import rearrange

import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding, BertModel, BertConfig, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from einops.layers.torch import Rearrange, EinMix
import einops

from .data import DataBatchPointer

import shared.constant
import shared.env

from utils.functional import repeat_init

logging.set_verbosity_error()


class ModelPointerEmbedder(Protocol):
    def get_feature_size(self) -> int:
        ...

    def set_main_device(self, value: torch.device):
        ...

    def __call__(self, input: DataBatchPointer) -> torch.Tensor:
        ...


class ModelPointerEmbedderPooling(nn.Module, ModelPointerEmbedder):
    def __init__(self) -> None:
        super().__init__()

        self.rearrange_in = Rearrange("batch step token -> (batch step) token")

        self.bert: BertModel = AutoModel.from_pretrained(
            shared.env.PRETRAINED_MODEL_POINTER
        )

        if not isinstance(self.bert, BertModel):
            raise NotImplementedError("Cannot use non-BERT models for embedding.")

        self.main_device = torch.device("cpu")

        for param in self.bert.encoder.layer[:9].parameters():  # type: ignore
            param.requires_grad_(False)
        for param in self.bert.embeddings.parameters():  # type: ignore
            param.requires_grad_(False)

    def get_feature_size(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, input: DataBatchPointer):
        pooler_output = cast(
            BaseModelOutputWithPoolingAndCrossAttentions,
            self.bert(
                **{k: self.rearrange_in(v) for k, v in input.get_input().items()}
            ),
        ).pooler_output
        """shape: ((batch step) feature)"""

        return einops.rearrange(
            pooler_output,
            "(batch step) feature -> batch step feature",
            batch=input.batch_size,
        ).to(self.main_device)

    def set_main_device(self, value: torch.device):
        self.main_device = value


class ModelPointerEmbedderSbert(nn.Module, ModelPointerEmbedder):
    def __init__(self):
        super().__init__()

        self.rearrange_in = Rearrange("batch step token -> (batch step) token")

        self.sbert = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )

    def get_feature_size(self) -> int:
        return self.sbert.config.hidden_size

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input: DataBatchPointer):
        model_input = {
            k: self.rearrange_in(v) for k, v in input.get_input().items()
        }  # type: ignore
        model_output = self.sbert(**model_input)

        pooling = self.mean_pooling(model_output, model_input["attention_mask"])

        result = einops.rearrange(
            pooling,
            "(batch step) feature -> batch step feature",
            batch=input.batch_size,
        ).to(self.main_device)

        return result

    def set_main_device(self, value: torch.device):
        self.main_device = value


@dataclass
class ModelOptionPointerLayer:
    feature: int = field(kw_only=True)
    """Embedding size."""

    ffn_dim: int = field(kw_only=True)
    """Dimension of the feed forward network."""

    attention_head: int = field(kw_only=True)
    """Attention head count."""

    dropout: float = field(kw_only=True, default=0.1)
    """Dropout."""


class FusionGate(nn.Module):
    def __init__(self, feature: int, batched: bool = True) -> None:
        super().__init__()

        self.feature = feature
        self.batched = batched

        self.z = nn.Sequential(
            Rearrange(
                "sum batch step feature -> batch step (sum feature)"
                if self.batched
                else "sum step feature -> step (sum feature)",
            ),
            nn.Linear(2 * feature, 1),
            nn.Sigmoid(),
        )

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor):
        if lhs.shape[-1] != self.feature or rhs.shape[-1] != self.feature:
            raise ValueError(
                f"Can not apply fusion gate to tensors of shape {lhs.shape} and {rhs.shape}, the last dim should be of size {self.feature}."
            )

        z = self.z(torch.stack([lhs, rhs]))
        result = z * lhs + (1.0 - z) * rhs
        return result


class ModelPointerEncoderLayer(nn.Module):
    def __init__(self, option: ModelOptionPointerLayer) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            option.feature,
            option.attention_head,
            dropout=option.dropout,
            batch_first=True,
        )
        # self.fusion_gate = FusionGate(option.feature)

        self.feed_forward = nn.Sequential(
            nn.Linear(option.feature, option.ffn_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(option.ffn_dim, option.feature),
            nn.Dropout(),
        )

        self.layer_norm1 = nn.LayerNorm(option.feature)
        self.layer_norm2 = nn.LayerNorm(option.feature)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.normal_(param)

    def forward(
        self,
        input: torch.Tensor,
        *,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.attention.num_heads, dim=0)

        x = input
        x = self.layer_norm1(
            x
            + self.attention(
                query=input,
                key=input,
                value=input,
                key_padding_mask=key_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )[0]
        )
        """shape(batch step feature)"""

        # x = x + self.fusion_gate(input, x)
        x = self.layer_norm2(x + self.feed_forward(x))
        """shape(batch step feature)"""

        return x


class ModelPointerDecoderLayer(nn.Module):
    def __init__(self, option: ModelOptionPointerLayer) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            option.feature,
            option.attention_head,
            dropout=option.dropout,
            batch_first=True,
        )
        # self.fusion_gate = FusionGate(option.feature)

        self.feed_forward = nn.Sequential(
            nn.Linear(option.feature, option.ffn_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(option.ffn_dim, option.feature),
            nn.Dropout(),
        )

        self.layer_norm1 = nn.LayerNorm(option.feature)
        self.layer_norm2 = nn.LayerNorm(option.feature)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.normal_(param)

    def forward(
        self,
        input: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.attention.num_heads, dim=0)

        x = input
        x = self.layer_norm1(
            x
            + self.attention(
                query=input,
                key=memory,
                value=memory,
                key_padding_mask=key_mask,
                attn_mask=attn_mask,
                need_weights=False,
            )[0]
        )

        x = self.layer_norm2(x + self.feed_forward(x))

        return x


@dataclass
class ModelPointerOption:
    embedder: ModelPointerEmbedder = field(kw_only=True)
    """The module that embeds the sentence."""

    # feature: int = field(kw_only=True)
    """Feature size"""

    encoder_layer: Callable[[], ModelPointerEncoderLayer] = field(kw_only=True)
    """Initializer of the encoder layer."""
    encoder_layer_count: int = field(kw_only=True)
    """How many layers in the encoder."""

    decoder_layer: Callable[[], ModelPointerDecoderLayer] = field(kw_only=True)
    """Initializer of the decoder layer."""
    decoder_layer_count: int = field(kw_only=True)
    """How many layers in the decoder."""

    # transform_adj_mat: Callable[[torch.Tensor], torch.Tensor] = field(kw_only=True)
    """How to transform the adjacent matrix."""


class ModelPointer(nn.Module):
    def __init__(self, option: ModelPointerOption) -> None:
        super().__init__()

        self.embedder = option.embedder
        self.feature = self.embedder.get_feature_size()

        self.encoder = nn.ModuleList(
            repeat_init(
                option.encoder_layer,
                option.encoder_layer_count,
            )
        )
        self.decoder = nn.ModuleList(
            repeat_init(
                option.decoder_layer,
                option.decoder_layer_count,
            )
        )

        self.ptr_embedding_dim = 2048

        self.ptr_w_q = nn.Parameter(
            torch.empty(self.feature)
        )  # pyright: ignore [reportPrivateImportUsage]
        self.ptr_w_k = nn.Parameter(
            torch.empty(self.feature)
        )  # pyright: ignore [reportPrivateImportUsage]

        nn.init.normal_(self.ptr_w_k)
        nn.init.normal_(self.ptr_w_q)

        self.prob = torch.sigmoid if shared.env.INSTRUCTION_IS_ORDERED else torch.tanh

        # self.ptr_activation = nn.functional.relu

    def forward(self, input: DataBatchPointer):
        embedding = self.embedder(input)

        batch, step, _ = embedding.shape

        key_mask = torch.zeros((batch, step), dtype=torch.bool, device=embedding.device)
        attn_mask = torch.zeros(
            (batch, step, step), dtype=torch.bool, device=embedding.device
        )
        for idx, (original_step, _) in enumerate(input.original_size):
            key_mask[idx, original_step:] = True
            attn_mask[idx, original_step:, :] = True
            attn_mask[idx, :, original_step:] = True

        memory = embedding
        for layer in self.encoder:
            memory = layer(memory, key_mask=key_mask)
        # memory = reduce(lambda i, layer: layer(i, key_mask=key_mask, attn_mask=attn_mask), self.encoder, embedding)

        # shape (batch step feature)

        # prev_state = memory
        def decoder_timestep(t: int):
            # nonlocal prev_state
            if t == 0:
                return torch.zeros((step, batch), device=embedding.device)

            decoder_input = embedding[..., t : t + 1, :]
            step_key_mask = torch.clone(key_mask)
            step_key_mask[..., t:] = True

            step_attn_mask = torch.zeros(
                (batch, 1, step), device=embedding.device, dtype=torch.bool
            )
            step_attn_mask[:, 0, t:] = True

            prev_state = reduce(
                lambda i, layer: layer(
                    i, memory, key_mask=step_key_mask, attn_mask=step_attn_mask
                ),
                self.decoder,
                decoder_input,
            )
            # (batch, one, feature)

            wq = einops.einsum(
                prev_state, self.ptr_w_q, "batch one feature, feature -> batch one"
            )[:, 0]

            ptrs = []
            for i in range(step):
                wk = einops.einsum(
                    memory[:, i], self.ptr_w_k, "batch feature, feature -> batch"
                )
                ptr = self.prob((wq + wk) / sqrt(self.feature))
                ptrs.append(ptr)

            return torch.stack(ptrs)

            # (ptrs, "timestep batch")

        output = einops.rearrange(
            [decoder_timestep(t) for t in range(step)],
            "step_query step_key batch -> batch step_key step_query",
        )

        # Q = einops.einsum(output, self.ptr_w_q, "batch step feature, feature -> batch step")
        # K = einops.einsum(memory, self.ptr_w_k, "batch step feature, feature -> batch step")

        # # P1 = einops.einsum(Q, K, "batch step_from, batch step_to -> batch step_from step_to") / sqrt(self.ptr_embedding_dim)

        # P1 = Q + K

        # P2 = torch.sigmoid(P1)

        # P3 = einops.rearrange(P2, "batch step_from step_to -> batch step_to step_from")
        # step_from depends on step_to.

        return output
