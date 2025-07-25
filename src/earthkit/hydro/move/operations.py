from earthkit.hydro.move import array
from earthkit.hydro.utils.decorators import xarray


@xarray
def upstream(river_network, field, node_weights=None, edge_weights=None, metric="sum"):
    return array.upstream(river_network, field, node_weights, edge_weights, metric)


@xarray
def downstream(
    river_network, field, node_weights=None, edge_weights=None, metric="sum"
):
    return array.downstream(river_network, field, node_weights, edge_weights, metric)
