from functools import wraps

from earthkit.hydro.backends import get_array_backend


def multi_backend(func):
    compiled_jax_fn = None

    @wraps(func)
    def wrapper(river_network, field, *args, **kwargs):
        xp = get_array_backend(field)
        backend_name = xp.name
        if backend_name == "jax":
            nonlocal compiled_jax_fn
            if compiled_jax_fn is None:
                from jax import jit

                def jax_func(field, *args, **kwargs):
                    return func(xp, river_network, field, *args, **kwargs)

                compiled_jax_fn = jit(jax_func)
            return compiled_jax_fn(field, *args, **kwargs)
        else:
            return func(xp, river_network, field, *args, **kwargs)

    return wrapper
