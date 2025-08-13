from earthkit.hydro._utils.decorators import mask, multi_backend
from earthkit.hydro._utils.locations import locations_to_1d
from earthkit.hydro.length.array import __operations as _operations


@multi_backend()
def min(xp, river_network, field, locations, upstream, downstream, return_grid):
    if field is None:
        field = xp.ones(river_network.n_nodes)
    locations, _, _ = locations_to_1d(xp, river_network, locations)
    decorated_func = mask(return_grid)(_operations.min)
    return decorated_func(xp, river_network, field, locations, upstream, downstream)


@multi_backend()
def max(xp, river_network, field, locations, upstream, downstream, return_grid):
    if field is None:
        field = xp.ones(river_network.n_nodes)
    locations, _, _ = locations_to_1d(xp, river_network, locations)
    decorated_func = mask(return_grid)(_operations.max)
    return decorated_func(xp, river_network, field, locations, upstream, downstream)


def to_source(*args, **kwargs):
    raise NotImplementedError


def to_sink(*args, **kwargs):
    raise NotImplementedError
