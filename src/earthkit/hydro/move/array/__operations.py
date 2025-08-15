from earthkit.hydro._core.move import calculate_move_metric


def upstream(xp, river_network, field, node_weights, edge_weights, metric):
    return calculate_move_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        flow_direction="up",
    )


def downstream(xp, river_network, field, node_weights, edge_weights, metric):
    return calculate_move_metric(
        xp,
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        flow_direction="down",
    )
