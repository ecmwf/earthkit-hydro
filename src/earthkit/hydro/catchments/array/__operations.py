from earthkit.hydro._core._find import _flow_find
from earthkit.hydro._utils.decorators import mask
from earthkit.hydro.upstream.array._operations import calculate_upstream_metric


def calculate_catchment_metric(
    xp,
    river_network,
    field,
    stations_1d,
    metric,
    node_weights,
    edge_weights,
):
    upstream_metric_field = calculate_upstream_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
    )
    return xp.gather(upstream_metric_field, stations_1d, axis=-1)


@mask(unmask=False)
def var(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "var",
        node_weights,
        edge_weights,
    )


@mask(unmask=False)
def std(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "std",
        node_weights,
        edge_weights,
    )


@mask(unmask=False)
def mean(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "mean",
        node_weights,
        edge_weights,
    )


@mask(unmask=False)
def sum(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "sum",
        node_weights,
        edge_weights,
    )


@mask(unmask=False)
def min(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "min",
        node_weights,
        edge_weights,
    )


@mask(unmask=False)
def max(xp, river_network, field, locations, node_weights, edge_weights):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "max",
        node_weights,
        edge_weights,
    )


def find(xp, river_network, field, overwrite, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_flow_find = mask(return_type == "gridded")(_flow_find)
    return decorated_flow_find(xp, river_network, field, overwrite)
