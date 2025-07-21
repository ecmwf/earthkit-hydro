import array_api_compat.cupy as cp

from .array_backend import ArrayBackend


class CuPyBackend(ArrayBackend):
    def __init__(self):
        super().__init__(cp)

    def copy(self, x):
        return x.copy()

    def scatter_add(self, target, indices, updates):
        cp.add.at(target, (*[slice(None)] * (target.ndim - 1), indices), updates)
        return target
