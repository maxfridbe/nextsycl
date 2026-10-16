class Data:
    """torch_geometric.data.Data: attributes, and .to() that keeps them (CPU only here)"""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def to(self, device):
        return self
