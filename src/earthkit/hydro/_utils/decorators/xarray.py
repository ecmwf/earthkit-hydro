from functools import wraps
from inspect import signature

import numpy as np
import xarray as xr

from earthkit.hydro._utils.coords import get_core_dims, node_default_coord


def get_full_signature(func, *args, **kwargs):
    sig = signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def assert_xr_compatible_backend(network):
    network_backend = network.array_backend
    if network_backend not in ["numpy", "cupy"]:
        raise NotImplementedError(f"xarray does not support {network_backend} backend")


def sort_xr_nonxr_args(all_args):
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
    return xr_args, non_xr_kwargs, arg_order


def get_reshuffled_func(func, arg_order):
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

    return reshuffled_func


def get_input_output_core_dims(
    input_core_dims, output_core_dims, xr_args, river_network, return_grid
):
    if input_core_dims is None:
        input_core_dims = [get_core_dims(xr_arg) for xr_arg in xr_args]
    elif len(input_core_dims) == 1:
        input_core_dims *= len(xr_args)

    if output_core_dims is None:
        if return_grid:
            if len(input_core_dims[0]) == 2:  # grid in and out
                output_core_dims = [input_core_dims[0]]
            else:
                output_core_dims = [
                    list(river_network.coords.keys())
                ]  # 1d in, grid out
        else:
            if len(input_core_dims[0]) == 1:  # 1d in and out
                output_core_dims = [input_core_dims[0]]
            else:
                output_core_dims = [[node_default_coord]]

    return input_core_dims, output_core_dims


def xarray(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Inspect the function signature and bind all arguments
        all_args = get_full_signature(func, *args, **kwargs)

        input_core_dims = all_args.pop("input_core_dims", None)
        output_core_dims = None

        assert_xr_compatible_backend(all_args["river_network"])

        # Separate xarray and non-xarray arguments
        xr_args, non_xr_kwargs, arg_order = sort_xr_nonxr_args(all_args)

        river_network = all_args["river_network"]
        return_type = all_args["return_type"]
        return_type = river_network.return_type if return_type is None else return_type
        return_grid = return_type == "gridded"

        if len(xr_args) == 0:
            output = func(**all_args)

            offset = 2 if return_grid else 1
            ndim = output.ndim
            dim_names = [f"axis{i + 1}" for i in range(ndim - offset)]
            coords = {
                dim: np.arange(size)
                for dim, size in zip(dim_names, output.shape[:-offset])
            }

            if return_grid:
                for k, v in river_network.coords.items():
                    coords[k] = v
                    dim_names.append(k)
            else:
                coords[node_default_coord] = np.arange(river_network.n_nodes)
                dim_names.append(node_default_coord)

            result = xr.DataArray(output, dims=dim_names, coords=coords, name="out")

            if not return_grid:
                coords_grid = np.meshgrid(*river_network.coords.values())
                assign_dict = {
                    k: (node_default_coord, v.flat[river_network.mask])
                    for k, v in zip(river_network.coords.keys(), coords_grid)
                }
                result = result.assign_coords(**assign_dict)
        else:

            reshuffled_func = get_reshuffled_func(func, arg_order)

            input_core_dims, output_core_dims = get_input_output_core_dims(
                input_core_dims, output_core_dims, xr_args, river_network, return_grid
            )

            output_sizes = (
                river_network.n_nodes
                if len(output_core_dims[0]) == 1
                else {k: v for k, v in zip(output_core_dims[0], river_network.shape)}
            )

            result = xr.apply_ufunc(
                reshuffled_func,
                *xr_args,
                input_core_dims=input_core_dims,
                output_core_dims=output_core_dims,
                dask_gufunc_kwargs={"output_sizes": output_sizes},
                output_dtypes=[float],
                dask="parallelized",
                kwargs=non_xr_kwargs,
            )

            if len(output_core_dims[0]) == 1:
                coords_grid = np.meshgrid(*river_network.coords.values())
                assign_dict = {
                    k: (output_core_dims[0], v.flat[river_network.mask])
                    for k, v in zip(river_network.coords.keys(), coords_grid)
                }
                result = result.assign_coords(**assign_dict)

        return result

    return wrapper
