from earthkit.hydro.catchments.array import _operations
from earthkit.hydro.utils.decorators import multi_backend


def preprocess_stations(xp, river_network, stations):
    if stations.ndim == 2 and stations.shape[1] == 2:
        if xp.name not in ["numpy", "cupy"]:
            raise NotImplementedError
        # TODO: make this code actually xp agnostic
        rows, cols = stations[:, 0], stations[:, 1]
        flat_indices = rows * river_network.shape[1] + cols
        flat_mask = river_network.mask
        reverse_map = -xp.ones(
            river_network.shape[0] * river_network.shape[1], dtype=int
        )
        reverse_map[flat_mask] = xp.arange(flat_mask.shape[0])
        masked_indices = reverse_map[flat_indices]
        if xp.any(masked_indices < 0):
            raise ValueError(
                "Some station points are not included in the masked array."
            )
        return masked_indices
    else:
        assert stations.ndim == 1
        return stations


@multi_backend(allow_jax_jit=False)
def var(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.var(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def std(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.std(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def mean(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.mean(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def sum(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.sum(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def min(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.min(
        river_network, field, stations_1d, node_weights, edge_weights
    )


@multi_backend(allow_jax_jit=False)
def max(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    locations = xp.asarray(locations, device=river_network.device)
    stations_1d = preprocess_stations(xp, river_network, locations)
    return _operations.max(
        river_network, field, stations_1d, node_weights, edge_weights
    )


def find(river_network, field):
    return _operations.find(river_network, field)
