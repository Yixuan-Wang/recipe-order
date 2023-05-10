from __future__ import annotations

from collections.abc import Iterable
import itertools

import torch
from torch import Tensor

from utils.functional import take


# @dataclass
# class TransformOriginal:
#     threshold: float = field(kw_only=True)

#     def simplify(self, mat: torch.Tensor) -> torch.Tensor:
#         adj = (mat > self.threshold).float()
#         adj = adj - adj.tril()

#         _ = itertools.accumulate(itertools.repeat(adj), torch.matmul, initial=adj)
#         _ = take(_, adj.shape[0])
#         one = (sum(_, start=torch.zeros_like(adj)) == 1).to(mat.dtype)

#         result = mat * one
#         return result

#     def __call__(self, mat: Tensor) -> Tensor:
#         return self.simplify(mat)


def threshold(threshold: float, mat: Tensor):
    return mat * (mat > threshold).to(mat.dtype)


def prune_matrix(adj: Tensor):
    adj = adj - adj.tril()
    _ = itertools.accumulate(
        itertools.repeat(adj), torch.matmul if adj.ndim == 2 else torch.bmm, initial=adj
    )
    _ = take(_, adj.shape[-2])
    one = (sum(_, start=torch.zeros_like(adj)) == 1).to(adj.dtype)

    return one


def build_graph_from_prediction(threshold: float, mat: Tensor):
    return prune_matrix((mat > threshold).to(mat.dtype))


def reachability_matrix(adj: Tensor):
    _ = itertools.accumulate(
        itertools.repeat(adj), torch.matmul if adj.ndim == 2 else torch.bmm, initial=adj
    )
    _ = take(_, adj.shape[-2])
    one = (sum(_, start=torch.zeros_like(adj)) != 0).to(adj.dtype)

    return one


def fold_matrix_to_triu(mat: Tensor):
    r"""
    Fold the matrix to upper triangle.

    For result matrix A'[i, j] where i < j, A'[i, j] == 1 means i must precede j, while A'[i, j] == -1 means j musty precede i and the original matrix A[j, i] == 1.
    """

    return mat - (tril := mat.tril()) - tril.mT


def unfold_matrix_from_triu(mat: Tensor):
    r"""
    Unfold the upper triangle.

    Only the upper triangle is populated in the input matrix A. For elements that A[i, j] == -1, in result matrix A', A'[i, j] will be 0 and A'[j, i] will be 1. A' is non-negative.
    """
    mat = mat - mat.tril()

    return ((ab := mat.abs()) + mat) / 2 + ((ab - mat) / 2).mT


def get_adj_matrix_from_edges(
    edges: Iterable[tuple[int, int]],
    *,
    graph_size: int,
) -> torch.Tensor:
    adj = torch.zeros((graph_size, graph_size))

    for x, y in edges:
        if x >= graph_size or y >= graph_size:
            raise ValueError(
                f"Edge <{x}, {y}> goes beyond the graph size of {graph_size}."
            )

        adj[x, y] = 1

    return adj
