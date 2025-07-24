def mask_and_unmask(func):

    def wrapper(xp, river_network, field, *args, **kwargs):

        # if field.shape[-2:] == river_network.shape:
        shape = field.shape
        B = shape[:-2]
        M, N = shape[-2], shape[-1]
        new_shape = B + (M * N,)
        field_flat = xp.reshape(field, new_shape)
        field_1d = xp.take_along_axis(field_flat, river_network.mask, axis=-1)
        out_1d = func(xp, river_network, field_1d, *args, **kwargs)
        out_flat = xp.full_like(field_flat, xp.nan)
        out_flat = xp.scatter_assign(out_flat, river_network.mask, out_1d)
        return xp.reshape(out_flat, shape)

    return wrapper
