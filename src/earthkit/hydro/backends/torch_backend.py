import array_api_compat.torch as torch

from .array_backend import ArrayBackend


class TorchBackend(ArrayBackend):
    def __init__(self):
        super().__init__(torch)

    def copy(self, x):
        return x.clone()

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        return target.index_add(-1, indices, updates)
