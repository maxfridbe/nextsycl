"""Readers for the two kinds of pickle UniMate's files are: a torch checkpoint (a ZIP holding data.pkl and the tensor
storages) and a numpy object array (a .npy holding a pickled dict). Neither is ever handed to pickle's own machinery:
the unpickler below answers every global with a stub of ours, so no callable of the file (or of numpy or torch) runs;
what it builds are plain dicts, lists, tuples and arrays read from the bytes.
"""
import collections
import io
import pickle
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64, "LongStorage": np.int64,
           "IntStorage": np.int32, "ShortStorage": np.int16, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
           "BFloat16Storage": None}


class _Stub:
    """What a global of the file becomes: it takes any arguments and state, and does nothing with them"""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):
        return _Stub(*args, **kwargs)


class _Storage:
    def __init__(self, kind, key, numel):
        self.kind, self.key, self.numel = kind, key, numel


class _TensorRec:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor(storage, offset, size, stride, *rest):
    return _TensorRec(storage, offset, size, stride)


def _make_storage_type(name):
    return type(name, (), {"__torch_storage__": name})


class _NDArrayMark:
    pass


class _dtype:
    def __init__(self, name, *rest):
        self.name = name

    def __setstate__(self, state):
        pass


class _Arr:
    """numpy's _reconstruct(ndarray, (0,), b'b') then __setstate__((version, shape, dtype, fortran, data))"""

    def __init__(self):
        self.value = None

    def __setstate__(self, st):
        _, shape, dt, fortran, data = st
        name = dt.name
        if isinstance(data, (bytes, bytearray)):
            a = np.frombuffer(bytes(data), dtype=np.dtype(name)).reshape(shape, order="F" if fortran else "C")
        else:
            a = np.array(data, dtype=object).reshape(shape)
        self.value = a


def _reconstruct(cls, shape, kind):
    return _Arr()


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return _make_storage_type(name)
        if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name == "_reconstruct":
            return _reconstruct
        if module == "numpy" and name == "ndarray":
            return _NDArrayMark
        if module == "numpy" and name == "dtype":
            return _dtype
        return _Stub

    def persistent_load(self, pid):
        # ('storage', <storage type>, key, location, numel)
        assert pid[0] == "storage", pid
        return _Storage(getattr(pid[1], "__torch_storage__", "?"), pid[2], pid[4])


def _unwrap(x):
    if isinstance(x, _Arr):
        v = x.value
        return _unwrap(v[()]) if v is not None and v.dtype == object and v.shape == () else v
    if isinstance(x, dict):
        return {k: _unwrap(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(v) for v in x)
    return x


def load_npy_dict(path):
    """A .npy holding a pickled dict (a 0-d object array): the dict, its arrays as numpy arrays"""
    b = open(path, "rb").read()
    assert b[:6] == b"\x93NUMPY", "not a .npy"
    ver = b[6]
    hl = int.from_bytes(b[8:10], "little") if ver == 1 else int.from_bytes(b[8:12], "little")
    start = (10 if ver == 1 else 12) + hl
    arr = _Unpickler(io.BytesIO(b[start:])).load()
    return _unwrap(arr)


def load_pth(path):
    """A torch checkpoint: (the zip, its root folder, the object it holds with every tensor a lazy record)"""
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"][0]
    root = pkl[: -len("data.pkl")]
    obj = _Unpickler(io.BytesIO(zf.read(pkl))).load()
    return zf, root, obj


def tensor(zf, root, rec):
    """The numpy array of a tensor record (float tensors only)"""
    st = rec.storage
    dt = _DTYPES[st.kind]
    assert dt is not None, st.kind
    raw = zf.read(f"{root}data/{st.key}")
    flat = np.frombuffer(raw, dtype=dt)
    item = flat.dtype.itemsize
    a = np.lib.stride_tricks.as_strided(flat[rec.offset:], shape=rec.size, strides=[s * item for s in rec.stride])
    return np.array(a)
