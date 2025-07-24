import jax.numpy as jnp

from .array_backend import ArrayBackend


class JAXBackend(ArrayBackend):
    def __init__(self):
        super().__init__(jnp)

    def copy(self, x):
        return x

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        return target.at[..., indices].set(updates)

    def scatter_add(self, target, indices, updates):
        return target.at[(*[slice(None)] * (target.ndim - 1), indices)].add(updates)
