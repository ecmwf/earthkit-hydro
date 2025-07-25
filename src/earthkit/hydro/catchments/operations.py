from earthkit.hydro.catchments import array
from earthkit.hydro.utils.decs.xarray import xarray


def name_last_dim(func):
    def wrapper(river_network, field, locations, *args, **kwargs):
        # TODO: transform locations into 2d array input
        if kwargs.get("output_core_dims", None) is None:
            kwargs["output_core_dims"] = [["station_index"]]
        if kwargs.get("output_sizes", None) is None:
            kwargs["output_sizes"] = {"station_index": locations.shape[0]}
        xr = func(river_network, field, locations, *args, **kwargs)
        # TODO: add nice coord info to xr
        # but first check xr is an xarray
        return xr

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
