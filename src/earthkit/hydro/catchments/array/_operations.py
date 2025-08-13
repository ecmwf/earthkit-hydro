from earthkit.hydro._utils.decorators import multi_backend
from earthkit.hydro.catchments.array import __operations as _operations

from ._utils import locations_to_1d


@multi_backend(allow_jax_jit=False)
def var(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.var(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def std(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.std(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def mean(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.mean(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def sum(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.sum(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def min(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.min(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def max(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.max(
        river_network, field, stations_1d, node_weights, edge_weights
    )


def find(river_network, locations):

    return _operations.find(river_network, locations)
