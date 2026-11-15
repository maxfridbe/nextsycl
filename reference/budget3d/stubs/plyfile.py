"""A stand-in for plyfile: Idol's gaussian.py imports it for save_ply, which ref.py does not call (it dumps .npy)."""


class PlyElement:
    @staticmethod
    def describe(*a, **k):
        raise NotImplementedError("ref.py dumps the Gaussians as .npy")


class PlyData:
    def __init__(self, *a, **k):
        raise NotImplementedError("ref.py dumps the Gaussians as .npy")
