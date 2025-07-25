from earthkit.hydro.core.move import calculate_move_metric
from earthkit.hydro.utils.decs import mask, multi_backend


@multi_backend()
@mask()
def upstream(
    xp, river_network, field, node_weights=None, edge_weights=None, metric="sum"
):
    return (
        calculate_move_metric(
            xp,
            river_network,
            field,
            metric,
            node_weights,
            edge_weights,
            flow_direction="up",
        )
        - field
    )


@multi_backend()
@mask()
def downstream(
    xp, river_network, field, node_weights=None, edge_weights=None, metric="sum"
):
    return (
        calculate_move_metric(
            xp,
            river_network,
            field,
            metric,
            node_weights,
            edge_weights,
            flow_direction="down",
        )
        - field
    )
