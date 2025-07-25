import numpy as np

from earthkit.hydro.upstream.array.operations import calculate_upstream_metric
from earthkit.hydro.utils.decs import mask, multi_backend


def calculate_catchment_metric(
    xp,
    river_network,
    field,
    stations_1d,
    metric,
    node_weights,
    edge_weights,
    mv,
    accept_missing,
):
    upstream_metric_field = calculate_upstream_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )
    return xp.gather(upstream_metric_field, stations_1d, axis=-1)


@multi_backend()
@mask(unmask=False)
def var(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "var",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend()
@mask(unmask=False)
def std(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "std",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend()
@mask(unmask=False)
def mean(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "mean",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend()
@mask(unmask=False)
def sum(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "sum",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


def find(*args, **kwargs):
    raise NotImplementedError
