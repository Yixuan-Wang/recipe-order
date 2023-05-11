from numpy import ones_like
import torch

def fit_topological_sort(adj: torch.Tensor):
    sub = adj - adj.T
    sub = sub * (sub > 0)

    mask = ~torch.ones_like(adj, dtype=torch.bool).triu()
    return 1 - sub.masked_select(mask).mean()
