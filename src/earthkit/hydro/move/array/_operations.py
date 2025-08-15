import earthkit.hydro.move.array.__operations as array
from earthkit.hydro._utils.decorators import mask, multi_backend


@multi_backend()
def upstream(xp, river_network, field, node_weights, edge_weights, metric, return_type):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_func = mask(return_type == "gridded")(array.upstream)
    return decorated_func(xp, river_network, field, node_weights, edge_weights, metric)


@multi_backend()
def downstream(
    xp, river_network, field, node_weights, edge_weights, metric, return_type
):
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    decorated_func = mask(return_type == "gridded")(array.downstream)
    return decorated_func(xp, river_network, field, node_weights, edge_weights, metric)
