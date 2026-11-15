"""A stand-in for tensordict's TensorDict, as much of it as Idol's and Budget3D's sampling code uses: a dict of
tensors with a batch size, `apply`, and `torch.cat` of several (ref.py puts this directory first on sys.path)."""
import torch


class TensorDict(dict):
    def __init__(self, d=None, batch_size=None, device=None):
        super().__init__(d or {})
        if batch_size is None:
            batch_size = []
        self.batch_size = torch.Size([batch_size] if isinstance(batch_size, int) else list(batch_size))

    def apply(self, fn):
        return TensorDict({k: fn(v) for k, v in self.items()}, batch_size=self.batch_size)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.cat:
            parts = args[0]
            dim = (kwargs or {}).get("dim", args[1] if len(args) > 1 else 0)
            out = {k: torch.cat([p[k] for p in parts], dim=dim) for k in parts[0].keys()}
            return TensorDict(out, batch_size=sum(p.batch_size[0] for p in parts))
        return NotImplemented
