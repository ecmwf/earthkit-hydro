import array_api_compat.cupy as cp

from .array_backend import ArrayBackend


class CuPyBackend(ArrayBackend):
    def __init__(self):
        super().__init__(cp)

    @property
    def name(self):
        return "cupy"

    def copy(self, x):
        return x.copy()

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        cp.add.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target

    def scatter_max(self, target, indices, updates):
        cp.maximum.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target

    def scatter_min(self, target, indices, updates):
        cp.minimum.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target
