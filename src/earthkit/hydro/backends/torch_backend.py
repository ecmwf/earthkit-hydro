import array_api_compat.torch as torch

from .array_backend import ArrayBackend


class TorchBackend(ArrayBackend):
    def __init__(self):
        super().__init__(torch)

    @property
    def name(self):
        return "torch"

    def copy(self, x):
        return x.clone()

    def gather(self, arr, indices, axis=-1):
        return torch.index_select(arr, dim=axis, index=indices)

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        return target.index_add(-1, indices, updates)

    def scatter_max(self, target, indices, updates):
        return target.index_reduce(-1, indices, updates, "amax")

    def scatter_min(self, target, indices, updates):
        return target.index_reduce(-1, indices, updates, "amin")
