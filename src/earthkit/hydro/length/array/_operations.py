from earthkit.hydro._utils.decorators import mask, multi_backend
from earthkit.hydro._utils.locations import locations_to_1d
from earthkit.hydro.length.array import __operations as _operations


@multi_backend()
def min(xp, river_network, field, locations, upstream, downstream, return_type):
    if field is None:
        field = xp.ones(river_network.n_nodes)
    locations, _, _ = locations_to_1d(xp, river_network, locations)
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_func = mask(return_type == "gridded")(_operations.min)
    return decorated_func(xp, river_network, field, locations, upstream, downstream)


@multi_backend()
def max(xp, river_network, field, locations, upstream, downstream, return_type):
    if field is None:
        field = xp.ones(river_network.n_nodes)
    locations, _, _ = locations_to_1d(xp, river_network, locations)
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_func = mask(return_type == "gridded")(_operations.max)
    return decorated_func(xp, river_network, field, locations, upstream, downstream)


def to_source(*args, **kwargs):
    raise NotImplementedError


def to_sink(*args, **kwargs):
    raise NotImplementedError
