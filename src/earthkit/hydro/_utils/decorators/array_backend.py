from functools import wraps

from earthkit.hydro._backends.find import get_array_backend


def multi_backend(allow_jax_jit=True, jax_static_args=None):
    def decorator(func):
        compiled_jax_fn = None

        @wraps(func)
        def wrapper(**kwargs):
            xp = get_array_backend(kwargs["river_network"].groups[0])
            backend_name = xp.name
            kwargs["xp"] = xp
            if backend_name == "jax" and allow_jax_jit:

                nonlocal compiled_jax_fn
                if compiled_jax_fn is None:
                    from jax import jit

                    compiled_jax_fn = jit(func, static_argnames=jax_static_args)
                return compiled_jax_fn(**kwargs)
            else:
                return func(**kwargs)

        return wrapper

    return decorator
