import earthkit.hydro.move.array.__operations as array
from earthkit.hydro._utils.decorators import mask, multi_backend


@multi_backend()
def upstream(xp, river_network, field, node_weights, edge_weights, metric, return_grid):
    decorated_func = mask(return_grid)(array.upstream)
    return decorated_func(xp, river_network, field, node_weights, edge_weights, metric)


@multi_backend()
def downstream(
    xp, river_network, field, node_weights, edge_weights, metric, return_grid
):
    decorated_func = mask(return_grid)(array.downstream)
    return decorated_func(xp, river_network, field, node_weights, edge_weights, metric)
