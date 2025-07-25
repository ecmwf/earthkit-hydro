from earthkit.hydro.downstream import array
from earthkit.hydro.utils.decorators import xarray


@xarray
def var(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return array.var(river_network, field, node_weights, edge_weights)


@xarray
def std(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return array.std(river_network, field, node_weights, edge_weights)


@xarray
def mean(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return array.mean(river_network, field, node_weights, edge_weights)


@xarray
def sum(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return array.sum(river_network, field, node_weights, edge_weights)


def min(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    raise NotImplementedError


def max(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    raise NotImplementedError
