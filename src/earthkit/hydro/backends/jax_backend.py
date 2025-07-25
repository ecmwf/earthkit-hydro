import jax
import jax.numpy as jnp

from .array_backend import ArrayBackend


class JAXBackend(ArrayBackend):
    def __init__(self):
        super().__init__(jnp)

    @property
    def name(self):
        return "jax"

    def copy(self, x):
        return x

    def gather(self, arr, indices, axis=-1):
        assert axis == -1
        return arr[..., indices]

    def scatter_assign(self, target, indices, updates):
        return target.at[..., indices].set(updates)

    def scatter_add(self, target, indices, updates):
        return target.at[(*[slice(None)] * (target.ndim - 1), indices)].add(updates)

    def asarray(self, arr, dtype=None, device=None, copy=None):
        for d in jax.devices():
            if d.platform == device:
                device = d
                break
        return jnp.asarray(arr, dtype=dtype, order=None, copy=copy, device=device)
