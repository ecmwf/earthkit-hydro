import numpy as np

from earthkit.hydro.core.online import calculate_online_metric
from earthkit.hydro.utils import xarray_mask_and_unmask
from earthkit.hydro.utils.decs import multi_backend


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


@xarray_mask_and_unmask
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


@xarray_mask_and_unmask
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


@xarray_mask_and_unmask
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


@xarray_mask_and_unmask
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


@xarray_mask_and_unmask
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


@multi_backend
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


@xarray_mask_and_unmask
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
