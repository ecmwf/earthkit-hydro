import numpy as np

from earthkit.hydro.core.online import calculate_online_metric
from earthkit.hydro.utils.decs import mask_and_unmask, multi_backend


def calculate_downstream_metric(
    xp,
    river_network,
    field,
    metric,
    node_weights,
    edge_weights,
    mv,
    accept_missing,
):
    return calculate_online_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
        flow_direction="up",
    )


@multi_backend
def var(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_downstream_metric(
        xp,
        river_network,
        field,
        "var",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend
def std(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_downstream_metric(
        xp,
        river_network,
        field,
        "std",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend
def mean(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_downstream_metric(
        xp,
        river_network,
        field,
        "mean",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )


@multi_backend
@mask_and_unmask
def sum(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_downstream_metric(
        xp,
        river_network,
        field,
        "sum",
        node_weights,
        edge_weights,
        mv=np.nan,
        accept_missing=False,
    )
