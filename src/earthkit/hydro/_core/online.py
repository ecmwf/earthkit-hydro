from .accumulate import flow
from .metrics import metrics_func_finder


def _calculate_mode_upstream(xp, river_network, field):
    """
    Calculate upstream mode using Rust implementation for performance.

    For categorical data, computes the most common (mode) value among all
    upstream nodes for each node in the river network.
    """
    # Mode only supported for numpy backend with Rust
    if xp.name != "numpy":
        raise NotImplementedError(
            "Mode calculation currently only supported for numpy backend"
        )

    # Import Rust function
    try:
        from earthkit.hydro import _rust
    except ImportError:
        raise ImportError(
            "Rust extension not available. Mode requires the Rust extension for performance."
        )

    # Convert field to int64 for categorical data
    import numpy as np
    field_int = np.asarray(field, dtype=np.int64)

    # Get network topology from sorted_data
    # sorted_data has shape (3, n_edges): [downstream_ids, upstream_ids, edge_ids]
    sorted_data = river_network._storage.sorted_data
    downstream_nodes = np.asarray(sorted_data[0, :], dtype=np.uintp)
    upstream_nodes = np.asarray(sorted_data[1, :], dtype=np.uintp)
    sources = np.asarray(river_network.sources, dtype=np.uintp)
    n_nodes = river_network.n_nodes

    # Call Rust function
    result = _rust.compute_upstream_mode_rust(
        field_int,
        upstream_nodes,
        downstream_nodes,
        sources,
        n_nodes
    )

    return xp.asarray(result)


def calculate_online_metric(
    xp,
    river_network,
    field,
    metric,
    node_weights,
    edge_weights,
    flow_direction,
):
    if flow_direction == "up":
        invert_graph = True
    elif flow_direction == "down":
        invert_graph = False
    else:
        raise ValueError(
            f"flow_direction must be 'up' or 'down', got {flow_direction}."
        )

    field = xp.copy(field)

    # Special handling for mode - uses Rust implementation
    if metric == "mode":
        if flow_direction != "down":
            raise ValueError("Mode calculation only supports upstream aggregation (flow_direction='down').")
        if node_weights is not None or edge_weights is not None:
            raise ValueError("Mode calculation does not support weights.")
        return _calculate_mode_upstream(xp, river_network, field)

    if node_weights is None:
        if metric == "mean" or metric == "std" or metric == "var":
            node_weights = xp.ones(river_network.n_nodes, dtype=xp.float64)
    else:
        node_weights = xp.copy(node_weights)

    if edge_weights is not None:
        edge_weights = xp.copy(edge_weights)

    func = metrics_func_finder(metric, xp).func

    weighted_field = flow(
        xp,
        river_network,
        field if node_weights is None else field * node_weights,
        func,
        invert_graph,
        edge_multiplicative_weight=edge_weights,
    )

    if metric == "mean" or metric == "std" or metric == "var":
        counts = flow(
            xp,
            river_network,
            xp.copy(node_weights),
            func,
            invert_graph,
            edge_multiplicative_weight=edge_weights,
        )

        if metric == "mean":
            weighted_field /= counts
            return weighted_field
        elif metric == "var" or metric == "std":
            weighted_sum_of_squares = flow(
                xp,
                river_network,
                field**2 if node_weights is None else field**2 * node_weights,
                func,
                invert_graph,
                edge_multiplicative_weight=edge_weights,
            )
            mean = weighted_field / counts
            weighted_sum_of_squares = weighted_sum_of_squares / counts - mean**2
            weighted_sum_of_squares = xp.clip(weighted_sum_of_squares, 0, xp.inf)
            if metric == "var":
                return weighted_sum_of_squares
            elif metric == "std":
                return xp.sqrt(weighted_sum_of_squares)
    else:
        return weighted_field
