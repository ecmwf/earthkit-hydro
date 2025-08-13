from functools import wraps
from inspect import signature

import xarray as xr


def xarray(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        input_core_dims = kwargs.pop("input_core_dims", None)
        output_core_dims = kwargs.pop("output_core_dims", None)
        output_sizes = kwargs.pop("output_sizes", None)

        # Introspect the function signature and bind all arguments
        sig = signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        all_args = bound_args.arguments

        # Separate xarray and non-xarray arguments
        xr_args = []
        non_xr_kwargs = {}
        arg_order = []

        for name, value in all_args.items():
            if isinstance(value, (xr.DataArray, xr.Dataset)):
                xr_args.append(value)
                arg_order.append(("xr", name))
            else:
                non_xr_kwargs[name] = value
                arg_order.append(("nonxr", name))

        # Create a function that reshuffles the args correctly
        def reshuffled_func(*only_xr_args, **non_xr_kwargs):
            full_args = {}
            xr_i = 0
            for kind, name in arg_order:
                if kind == "xr":
                    full_args[name] = only_xr_args[xr_i]
                    xr_i += 1
                else:
                    full_args[name] = non_xr_kwargs[name]
            return func(**full_args)

        input_core_dims = (
            [["lat", "lon"]] * len(xr_args)
            if input_core_dims is None
            else (
                input_core_dims * len(xr_args)
                if len(input_core_dims) == 1
                else input_core_dims
            )
        )
        output_core_dims = (
            [["lat", "lon"]] if output_core_dims is None else output_core_dims
        )

        return xr.apply_ufunc(
            reshuffled_func,
            *xr_args,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            dask_gufunc_kwargs={"output_sizes": output_sizes},
            output_dtypes=[float],
            dask="parallelized",
            kwargs=non_xr_kwargs,
        )

    return wrapper
