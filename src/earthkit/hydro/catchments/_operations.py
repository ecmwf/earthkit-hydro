from functools import wraps

import xarray as xr

import earthkit.hydro.catchments.array.__operations as array
from earthkit.hydro._backends.find import get_array_backend
from earthkit.hydro._utils.decorators import multi_backend
from earthkit.hydro.catchments.array._utils import locations_to_1d


def _name_last_dim(func):
    @wraps(func)
    def wrapper(river_network, field, locations, *args, **kwargs):

        xp = get_array_backend(river_network.groups[0])

        stations_1d, locations, orig_locations = locations_to_1d(
            xp, river_network, locations
        )

        if kwargs.get("output_core_dims", None) is None:
            kwargs["output_core_dims"] = [["station_index"]]
        if kwargs.get("output_sizes", None) is None:
            kwargs["output_sizes"] = {"station_index": locations.shape[0]}

        result = func(river_network, field, stations_1d, *args, **kwargs)

        if isinstance(field, xr.Dataset) or isinstance(field, xr.DataArray):
            if isinstance(orig_locations, dict):
                names = xp.asarray(
                    list(orig_locations.keys()), device=river_network.device
                )
                coords = xp.asarray(
                    list(orig_locations.values()), device=river_network.device
                )
                result = result.assign_coords(
                    station_name=("station_index", names),
                    lat=("station_index", coords[:, 0]),
                    lon=("station_index", coords[:, 1]),
                )

            if locations.ndim == 1:
                result = result.assign_coords(idx=("station_index", locations))
            elif locations.ndim == 2:
                # TODO: check if xp agnostic
                result = result.assign_coords(
                    xidx=("station_index", locations[:, 0]),
                    yidx=("station_index", locations[:, 1]),
                )
        return result

    return wrapper


@multi_backend(allow_jax_jit=False)
def var(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.var(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def std(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.std(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def mean(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.mean(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def sum(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.sum(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def min(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.min(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def max(
    xp,
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.max(xp, river_network, field, locations, node_weights, edge_weights)


@multi_backend(allow_jax_jit=False)
def find(xp, river_network, field):
    array.find(xp, river_network, field)
