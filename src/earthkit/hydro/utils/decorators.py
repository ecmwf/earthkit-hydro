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

    def wrapper(*args, **kwargs):
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
        river_network = kwargs["river_network"]

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

        return func(*args, **kwargs)

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

    def wrapper(field, *args, **kwargs):
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

        river_network = kwargs["river_network"]

        if field.shape[-2:] == river_network.shape:

            values_on_river_network = mask_2d(func)(field, *args, **kwargs)

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
            return mask_2d(func)(field, *args, **kwargs)

    return wrapper


def xarray_mask_and_unmask(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(a, (xr.DataArray, xr.Dataset)) for a in args) or any(
            isinstance(a, (xr.DataArray, xr.Dataset)) for a in kwargs.values()
        ):
            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_args = bound_args.arguments

            xr_args = []
            non_xr_values = {}
            for name, value in all_args.items():
                if isinstance(value, (xr.DataArray, xr.Dataset)):
                    xr_args.append(value)
                else:
                    non_xr_values[name] = value

            def wrapped_func(*only_xr_args, **kwargs):
                full_args = []
                i = 0
                for name in all_args:
                    if name in non_xr_values:
                        full_args.append(kwargs[name])
                    else:
                        full_args.append(only_xr_args[i])
                        i += 1
                return func(*full_args)

            # TODO: Avoid hard coding lat and lon as coord names...
            return xr.apply_ufunc(
                mask_and_unmask(wrapped_func),
                *xr_args,
                input_core_dims=[["lat", "lon"] * len(xr_args)],
                output_core_dims=[["lat", "lon"]],
                output_dtypes=[float],
                vectorize=True,
                dask="parallelized",
                kwargs=non_xr_values,
            )
        else:
            return mask_and_unmask(func)(*args, **kwargs)

    return wrapper
