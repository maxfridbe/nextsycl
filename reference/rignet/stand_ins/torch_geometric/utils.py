import torch


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], None if edge_attr is None else edge_attr[mask]


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    n = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    loop = torch.arange(n, dtype=edge_index.dtype)
    return torch.cat([edge_index, torch.stack([loop, loop])], dim=1), None


def softmax(src, index, num_nodes=None):
    raise NotImplementedError("not used by RigNet's inference")
