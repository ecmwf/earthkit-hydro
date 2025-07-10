import numpy as np

from earthkit.hydro.utils import missing_to_nan, nan_to_missing

from .accumulate import flow
from .metrics import metrics_dict


def calculate_online_metric(
    river_network,
    field,
    metric,
    node_weights,
    edge_weights,
    mv,
    accept_missing,
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

    field, field_dtype = missing_to_nan(field.copy(), mv, accept_missing)

    if node_weights is None:
        if metric == "mean" or metric == "std" or metric == "var":
            node_weights = np.ones(river_network.n_nodes, dtype=np.float64)
    else:
        if field_dtype != node_weights.dtype:
            raise ValueError(
                f"""
                node_weights.dtype={node_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        node_weights, _ = missing_to_nan(node_weights.copy(), mv, accept_missing)

    if edge_weights is not None:
        if field_dtype != edge_weights.dtype:
            raise ValueError(
                f"""
                edge_weights.dtype={edge_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        edge_weights, _ = missing_to_nan(edge_weights.copy(), mv, accept_missing)

    ufunc = metrics_dict[metric].func

    weighted_field = flow(
        river_network,
        field if node_weights is None else field * node_weights,
        ufunc,
        invert_graph,
        edge_multiplicative_weight=edge_weights,
    )

    if metric == "mean" or metric == "std" or metric == "var":
        counts = flow(
            river_network,
            node_weights.copy(),
            ufunc,
            invert_graph,
            edge_multiplicative_weight=edge_weights,
        )

        if metric == "mean":
            weighted_field /= counts  # weighted mean
            return nan_to_missing(
                weighted_field, np.float64, mv
            )  # if we compute means, we change dtype for int fields etc.
        elif metric == "var" or metric == "std":
            weighted_sum_of_squares = flow(
                river_network,
                field**2 if node_weights is None else field**2 * node_weights,
                ufunc,
                invert_graph,
                edge_multiplicative_weight=edge_weights,
            )
            mean = weighted_field / counts
            weighted_sum_of_squares = weighted_sum_of_squares / counts - mean**2
            weighted_sum_of_squares[weighted_sum_of_squares < 0] = (
                0  # can occur for numerical issues
            )
            if metric == "var":
                return nan_to_missing(weighted_sum_of_squares, np.float64, mv)
            elif metric == "std":
                return nan_to_missing(np.sqrt(weighted_sum_of_squares), np.float64, mv)

    else:
        return nan_to_missing(weighted_field, field_dtype, mv)
