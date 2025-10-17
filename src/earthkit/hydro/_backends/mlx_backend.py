import mlx.core as mx

from .array_backend import ArrayBackend


class MLXBackend(ArrayBackend):
    def __init__(self):
        super().__init__(mx)

    @property
    def name(self):
        return "mlx"

    def copy(self, x):
        return x

    def asarray(self, x, *args, **kwargs):
        return mx.array(x)

    def full(self, *args, **kwargs):
        kwargs.pop("device")
        return mx.full(*args, **kwargs)

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        target[..., indices] = updates
        return target

    def scatter_add(self, target, indices, updates):
        return target.at[..., indices].add(updates)
