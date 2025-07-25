from earthkit.hydro.core.online import calculate_online_metric
from earthkit.hydro.utils.decs import mask, multi_backend


def calculate_upstream_metric(
    xp,
    river_network,
    field,
    metric,
    node_weights,
    edge_weights,
):
    return calculate_online_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        flow_direction="down",
    )


@multi_backend()
@mask()
def var(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_upstream_metric(
        xp,
        river_network,
        field,
        "var",
        node_weights,
        edge_weights,
    )


@multi_backend()
@mask()
def std(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_upstream_metric(
        xp,
        river_network,
        field,
        "std",
        node_weights,
        edge_weights,
    )


@multi_backend()
@mask()
def mean(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_upstream_metric(
        xp,
        river_network,
        field,
        "mean",
        node_weights,
        edge_weights,
    )


@multi_backend()
@mask()
def sum(
    xp,
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    return calculate_upstream_metric(
        xp,
        river_network,
        field,
        "sum",
        node_weights,
        edge_weights,
    )


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
