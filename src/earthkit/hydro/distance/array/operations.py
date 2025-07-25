from earthkit.hydro.catchments.array.operations import preprocess_stations
from earthkit.hydro.distance.array import _operations
from earthkit.hydro.utils.decorators import multi_backend


@multi_backend()
def min(xp, river_network, field, locations, upstream=False, downstream=True):
    locations = xp.asarray(locations, device=river_network.device)
    locations = preprocess_stations(xp, river_network, locations)
    # TODO: add back when we have default xarray with correct lat lon
    # if weights is None:
    #     weights = xp.ones(river_network.shape)
    return _operations.min(xp, river_network, field, locations, upstream, downstream)


@multi_backend()
def max(xp, river_network, field, locations, upstream=False, downstream=True):
    locations = xp.asarray(locations, device=river_network.device)
    locations = preprocess_stations(xp, river_network, locations)
    # TODO: add back when we have default xarray with correct lat lon
    # if weights is None:
    #     weights = xp.ones(river_network.shape)
    return _operations.max(xp, river_network, field, locations, upstream, downstream)


def to_source(*args, **kwargs):
    raise NotImplementedError


def to_sink(*args, **kwargs):
    raise NotImplementedError
