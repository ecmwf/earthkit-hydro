from functools import wraps
from inspect import signature

import numpy as np
import xarray as xr


def mask_2d(func):
    """Decorator to allow function to mask 2d inputs to the river network.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.

    """

    def wrapper(river_network, *args, **kwargs):
        """Wrapper masking 2d data fields to allow for processing along the
        river network, then undoing the masking.

        Parameters
        ----------
        river_network : object
            The RiverNetwork instance calling the method.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.

        """

        args = tuple(
            (
                arg[..., river_network.mask]
                if isinstance(arg, np.ndarray) and arg.shape[-2:] == river_network.shape
                else arg if isinstance(arg, np.ndarray) else arg
            )
            for arg in args
        )

        kwargs = {
            key: (
                value[..., river_network.mask]
                if isinstance(value, np.ndarray)
                and value.shape[-2:] == river_network.shape
                else value if isinstance(value, np.ndarray) else value
            )
            for key, value in kwargs.items()
        }

        return func(river_network, *args, **kwargs)

    return wrapper


def mask_and_unmask(func):
    """Decorator to convert masked 2d inputs back to 1d.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.

    """

    @wraps(func)
    def wrapper(river_network, field, *args, **kwargs):
        """Wrapper masking 2d data fields to allow for processing along the
        river network, then undoing the masking.

        Parameters
        ----------
        field : numpy.ndarray
            The input data field to be processed.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.

        """

        # gets the missing value from the keyword arguments if it is present,
        # otherwise takes default value of mv from func
        mv = kwargs.get("mv")
        mv = mv if mv is not None else func.__defaults__[0]

        if field.shape[-2:] == river_network.shape:

            values_on_river_network = mask_2d(func)(
                river_network, field, *args, **kwargs
            )

            out_field = np.empty(field.shape, dtype=values_on_river_network.dtype)

            out_field[..., river_network.mask] = values_on_river_network

            if np.result_type(mv, field) != field.dtype:
                raise ValueError(
                    f"Missing value of type {type(mv)} is not compatible"
                    f" with field of dtype {field.dtype}"
                )

            out_field[..., ~river_network.mask] = mv
            return out_field
        else:
            return mask_2d(func)(river_network, field, *args, **kwargs)

    return wrapper


def xarray_mask_and_unmask(func):
    func = mask_and_unmask(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Fast path: no xarray inputs, use original logic
        if not (
            any(isinstance(a, (xr.DataArray, xr.Dataset)) for a in args)
            or any(isinstance(a, (xr.DataArray, xr.Dataset)) for a in kwargs.values())
        ):
            return func(*args, **kwargs)

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
        def reshuffled_func(*only_xr_args, **kwargs):
            full_args = {}
            xr_i = 0
            for kind, name in arg_order:
                if kind == "xr":
                    full_args[name] = only_xr_args[xr_i]
                    xr_i += 1
                else:
                    full_args[name] = kwargs[name]
            return func(**full_args)

        # TODO: Avoid hardcoding lat/lon
        input_core_dims = [["lat", "lon"]] * len(xr_args)

        return xr.apply_ufunc(
            reshuffled_func,
            *xr_args,
            input_core_dims=input_core_dims,
            output_core_dims=[["lat", "lon"]],
            output_dtypes=[float],
            vectorize=True,
            dask="parallelized",
            kwargs=non_xr_kwargs,
        )

    return wrapper
