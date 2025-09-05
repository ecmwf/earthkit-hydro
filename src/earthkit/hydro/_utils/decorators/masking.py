from functools import wraps


def mask(unmask=True):

    def decorator(func):

        @wraps(func)
        def wrapper(xp, river_network, field, *args, **kwargs):

            if field.shape[-2:] == river_network.shape:
                args, kwargs = process_args_kwargs(xp, river_network, args, kwargs)
                field_1d = mask_last2_dims(xp, field, river_network.mask, field.shape)

                out_1d = func(xp, river_network, field_1d, *args, **kwargs)

                if unmask:
                    out_shape = field.shape
                    return scatter_and_reshape(
                        xp, river_network.mask, out_1d, out_shape
                    )
                else:
                    return out_1d
            else:
                out_1d = func(xp, river_network, field, *args, **kwargs)
                if unmask:
                    out_shape = field.shape[:-2] + river_network.shape
                    return scatter_and_reshape(
                        xp, river_network.mask, out_1d, out_shape
                    )
                else:
                    return out_1d

        return wrapper

    return decorator


def mask_last2_dims(xp, tensor, mask, target_shape):
    B = target_shape[:-2]
    M, N = target_shape[-2], target_shape[-1]
    flat_shape = B + (M * N,)
    tensor_flat = xp.reshape(tensor, flat_shape)
    return xp.gather(tensor_flat, mask, axis=-1)


def scatter_and_reshape(xp, mask, out_1d, target_shape):
    B = target_shape[:-2]
    M, N = target_shape[-2], target_shape[-1]
    flat_shape = B + (M * N,)
    out_flat = xp.full(flat_shape, xp.nan)
    out_flat = xp.scatter_assign(out_flat, mask, out_1d)
    return xp.reshape(out_flat, target_shape)


def process_args_kwargs(xp, river_network, args, kwargs):
    def process_arg(arg):
        if (
            hasattr(arg, "shape")  # TODO: decide if robust enough
            and len(arg.shape) >= 2
            and arg.shape[-2:] == river_network.shape
        ):
            return mask_last2_dims(xp, arg, river_network.mask, arg.shape)
        return arg

    new_args = tuple(process_arg(arg) for arg in args)
    new_kwargs = {k: process_arg(v) for k, v in kwargs.items()}
    return new_args, new_kwargs
