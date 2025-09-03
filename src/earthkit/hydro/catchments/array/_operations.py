from earthkit.hydro._utils.decorators import multi_backend
from earthkit.hydro._utils.locations import locations_to_1d
from earthkit.hydro.catchments.array import __operations as _operations


@multi_backend(allow_jax_jit=False)
def var(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.var(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def std(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.std(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def mean(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.mean(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def sum(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.sum(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def min(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.min(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def max(xp, river_network, field, locations, node_weights, edge_weights):
    stations_1d, _, _ = locations_to_1d(xp, river_network, locations)
    return _operations.max(
        xp, river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend()
def find(xp, river_network, locations, overwrite, return_type):
    stations1d, _, _ = locations_to_1d(xp, river_network, locations)
    field = xp.full(river_network.n_nodes, xp.nan, device=river_network.device)
    field[stations1d] = xp.arange(stations1d.shape[0])
    return _operations.find(xp, river_network, field, overwrite, return_type)
