import array_api_compat.numpy as np

from .array_backend import ArrayBackend


class NumPyBackend(ArrayBackend):
    def __init__(self):
        super().__init__(np)

    @property
    def name(self):
        return "numpy"

    def copy(self, x):
        return x.copy()

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        np.add.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target

    def scatter_max(self, target, indices, updates):
        np.maximum.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target

    def scatter_min(self, target, indices, updates):
        np.minimum.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target
