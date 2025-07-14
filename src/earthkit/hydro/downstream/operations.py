import numpy as np

from earthkit.hydro.core.online import calculate_online_metric
from earthkit.hydro.utils import xarray_mask_and_unmask


@xarray_mask_and_unmask
def calculate_downstream_metric(
    field,
    node_weights,
    edge_weights,
    river_network,
    metric,
    mv,
    accept_missing,
):
    return calculate_online_metric(
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
        flow_direction="up",
    )


def var(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        field=field,
        node_weights=node_weights,
        edge_weights=edge_weights,
        river_network=river_network,
        metric="var",
        mv=mv,
        accept_missing=accept_missing,
    )


def std(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "std",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def mean(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "mean",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def min(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "min",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def max(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "max",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def sum(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "sum",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def prod(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_downstream_metric(
        river_network,
        field,
        "prod",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )
