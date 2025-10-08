import dask.array as da

from .array_backend import ArrayBackend


class DaskBackend(ArrayBackend):
    def __init__(self):
        super().__init__(da)

    @property
    def name(self):
        return "dask"

    def copy(self, x):
        return x

    def asarray(self, x, *args, **kwargs):
        return da.asarray(x)

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        target[..., indices] += updates
        return target
