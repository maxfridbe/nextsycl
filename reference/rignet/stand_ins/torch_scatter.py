"""torch_scatter's scatter_max / scatter_mean along dim 0, in pure torch (a stand-in for reference/rignet/ref.py)"""
import torch


def scatter_max(src, index, dim=0):
    assert dim == 0
    n = int(index.max()) + 1
    out = torch.zeros(n, src.size(1), dtype=src.dtype).scatter_reduce(0, index[:, None].expand_as(src), src, reduce="amax", include_self=False)
    return out, None


def scatter_mean(src, index, dim=0):
    assert dim == 0
    n = int(index.max()) + 1
    out = torch.zeros(n, src.size(1), dtype=src.dtype).index_add(0, index, src)
    cnt = torch.zeros(n, dtype=src.dtype).index_add(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt[:, None]
