"""MessagePassing (max aggregation over x only), PointConv (PointNetConv 1.7), and torch_cluster's fps / radius / knn as
PyTorch Geometric 1.7 wraps them - except fps, which starts at the first point here (torch_cluster's default starts
at a random one), as the nextsycl port does."""
import math

import torch

from .utils import add_self_loops, remove_self_loops


def scatter_amax(msg, dst, n):
    out = torch.zeros(n, msg.size(1), dtype=msg.dtype)
    return out.scatter_reduce(0, dst[:, None].expand_as(msg), msg, reduce="amax", include_self=False)


class MessagePassing(torch.nn.Module):
    def __init__(self, aggr="max", **kw):
        super().__init__()
        assert aggr == "max"

    def propagate(self, edge_index, x):
        src, dst = edge_index[0], edge_index[1]
        msg = self.message(x[dst], x[src])
        return self.update(scatter_amax(msg, dst, x.size(0)))

    def update(self, aggr_out):
        return aggr_out


class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None, **kw):
        super().__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, x, pos, edge_index):
        if not isinstance(x, tuple):
            x = (x, None)
        if not isinstance(pos, tuple):
            pos = (pos, pos)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=pos[1].size(0))
        src, dst = edge_index[0], edge_index[1]
        msg = pos[0][src] - pos[1][dst]
        if x[0] is not None:
            msg = torch.cat([x[0][src], msg], dim=1)
        msg = self.local_nn(msg)
        out = scatter_amax(msg, dst, pos[1].size(0))
        return out if self.global_nn is None else self.global_nn(out)


def fps(pos, batch=None, ratio=0.5, random_start=False):
    n = pos.size(0)
    k = int(math.ceil(float(torch.tensor(n, dtype=torch.float32) * torch.tensor(ratio, dtype=torch.float32))))
    k = max(1, min(k, n))
    out = [0]
    dist = ((pos - pos[0]) ** 2).sum(1)
    while len(out) < k:
        i = int(torch.argmax(dist))
        out.append(i)
        dist = torch.minimum(dist, ((pos - pos[i]) ** 2).sum(1))
    return torch.tensor(out, dtype=torch.long)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    rows, cols = [], []
    for i in range(y.size(0)):
        d = ((x - y[i]) ** 2).sum(1)
        cand = [(float(d[j]), j) for j in range(x.size(0)) if float(d[j]) < r * r]
        cand.sort()
        for _, j in cand[:max_num_neighbors]:
            rows.append(i)
            cols.append(j)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(cols, dtype=torch.long)


def global_max_pool(x, batch):
    return x.max(dim=0, keepdim=True).values


def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    out = []
    for i in range(pos_y.size(0)):
        d = ((pos_x - pos_y[i]) ** 2).sum(1)
        order = sorted(range(pos_x.size(0)), key=lambda j: (float(d[j]), j))[:k]
        w = torch.stack([1.0 / torch.clamp(d[j], min=1e-16) for j in order])
        out.append((w[:, None] * x[order]).sum(0) / w.sum())
    return torch.stack(out)
