import numpy as np
import xarray as xr

import earthkit.hydro.catchments.array._operations as array
from earthkit.hydro.backends.find import get_array_backend
from earthkit.hydro.catchments.array.operations import preprocess_stations
from earthkit.hydro.utils.decorators.xarray import xarray


def name_last_dim(func):
    def wrapper(river_network, field, locations, *args, **kwargs):
        orig_locations = locations
        xr_present = isinstance(field, (xr.DataArray, xr.Dataset))
        dict_locations = isinstance(locations, dict)
        if dict_locations:
            assert xr_present

            lats = field.lat.data
            lons = field.lon.data

            # TODO: decide if acceptable to always use np here
            locations = []
            for lat_val, lon_val in orig_locations.values():
                ilat = np.abs(lats - lat_val).argmin()
                ilon = np.abs(lons - lon_val).argmin()
                locations.append((int(ilat), int(ilon)))

        xp = get_array_backend(river_network.groups[0])
        locations = xp.asarray(locations, device=river_network.device)
        stations_1d = preprocess_stations(xp, river_network, locations)

        if kwargs.get("output_core_dims", None) is None:
            kwargs["output_core_dims"] = [["station_index"]]
        if kwargs.get("output_sizes", None) is None:
            kwargs["output_sizes"] = {"station_index": locations.shape[0]}

        result = func(river_network, field, stations_1d, *args, **kwargs)

        if xr_present:
            if dict_locations:
                # TODO: decide if should be xr and not list
                names = list(orig_locations.keys())
                result = result.assign_coords(
                    station_name=("station_index", names),
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


@name_last_dim
@xarray
def var(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.var(river_network, field, locations, node_weights, edge_weights)


@name_last_dim
@xarray
def std(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.std(river_network, field, locations, node_weights, edge_weights)


@name_last_dim
@xarray
def mean(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.mean(river_network, field, locations, node_weights, edge_weights)


@name_last_dim
@xarray
def sum(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    return array.sum(river_network, field, locations, node_weights, edge_weights)


def find(*args, **kwargs):
    raise NotImplementedError


def min(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    raise NotImplementedError


def max(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    raise NotImplementedError
