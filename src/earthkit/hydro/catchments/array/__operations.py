from earthkit.hydro._core._find import _flow_find
from earthkit.hydro._utils.decorators import mask, multi_backend
from earthkit.hydro.catchments.array._utils import locations_to_1d
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
def var(xp, river_network, field, locations, node_weights=None, edge_weights=None):
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
def std(xp, river_network, field, locations, node_weights=None, edge_weights=None):
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
def mean(xp, river_network, field, locations, node_weights=None, edge_weights=None):
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
def sum(xp, river_network, field, locations, node_weights=None, edge_weights=None):
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
def min(xp, river_network, field, locations, node_weights=None, edge_weights=None):
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
def max(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    return calculate_catchment_metric(
        xp,
        river_network,
        field,
        locations,
        "max",
        node_weights,
        edge_weights,
    )


@multi_backend()
def find(xp, river_network, locations):
    stations1d, _, _ = locations_to_1d(xp, river_network, locations)
    field = xp.full(river_network.n_nodes, xp.nan, device=river_network.device)
    field[stations1d] = xp.arange(stations1d.shape[0])
    return _flow_find(xp, river_network, field)
