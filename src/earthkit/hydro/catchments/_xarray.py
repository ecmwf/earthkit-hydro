from functools import wraps

import numpy as np
import xarray as xr

from earthkit.hydro._backends.find import get_array_backend
from earthkit.hydro._utils.coords import get_core_dims, node_default_coord
from earthkit.hydro._utils.decorators.xarray import (
    assert_xr_compatible_backend,
    get_full_signature,
    get_reshuffled_func,
    sort_xr_nonxr_args,
)
from earthkit.hydro._utils.locations import locations_to_1d


def get_input_core_dims(input_core_dims, xr_args):
    if input_core_dims is None:
        input_core_dims = [get_core_dims(xr_arg) for xr_arg in xr_args]
    elif len(input_core_dims) == 1:
        input_core_dims *= len(xr_args)

    return input_core_dims


def xarray(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Inspect the function signature and bind all arguments
        all_args = get_full_signature(func, *args, **kwargs)

        input_core_dims = all_args.pop("input_core_dims", None)

        assert_xr_compatible_backend(all_args["river_network"])

        river_network = all_args["river_network"]

        xp = get_array_backend(river_network.array_backend)

        locations = all_args["locations"]

        stations_1d, locations, orig_locations = locations_to_1d(
            xp, river_network, locations
        )

        all_args["locations"] = stations_1d

        # Separate xarray and non-xarray arguments
        xr_args, non_xr_kwargs, arg_order = sort_xr_nonxr_args(all_args)

        if len(xr_args) == 0:
            output = func(**all_args)

            ndim = output.ndim
            dim_names = [f"axis{i + 1}" for i in range(ndim - 1)]
            coords = {
                dim: np.arange(size) for dim, size in zip(dim_names, output.shape[:-1])
            }

            coords[node_default_coord] = np.arange(river_network.n_nodes)[stations_1d]
            dim_names.append(node_default_coord)

            result = xr.DataArray(output, dims=dim_names, coords=coords, name="out")

        else:

            reshuffled_func = get_reshuffled_func(func, arg_order)

            input_core_dims = get_input_core_dims(input_core_dims, xr_args)

            result = xr.apply_ufunc(
                reshuffled_func,
                *xr_args,
                input_core_dims=input_core_dims,
                output_core_dims=[[node_default_coord]],
                dask_gufunc_kwargs={"output_sizes": stations_1d.shape[0]},
                output_dtypes=[float],
                dask="parallelized",
                kwargs=non_xr_kwargs,
            )
            assign_dict = {
                node_default_coord: (
                    node_default_coord,
                    np.arange(river_network.n_nodes)[stations_1d],
                )
            }
            result = result.assign_coords(**assign_dict)

        coord1, coord2 = river_network.coords.values()
        coords_grid = np.meshgrid(coord2, coord1)
        assign_dict = {
            k: (node_default_coord, v.flat[river_network.mask][stations_1d])
            for k, v in zip(river_network.coords.keys(), coords_grid)
        }
        if isinstance(orig_locations, dict):
            assign_dict["name"] = (node_default_coord, list(orig_locations.keys()))
        result = result.assign_coords(**assign_dict)

        return result

    return wrapper
