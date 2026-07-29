# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from ._move import move_python as flow
from .metrics import metrics_func_finder


def calculate_move_metric(
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
        node_modifier_use_upstream = True
    elif flow_direction == "down":
        invert_graph = False
        node_modifier_use_upstream = True
    else:
        raise ValueError(f"flow_direction must be 'up' or 'down', got {flow_direction}.")

    if node_weights is None:
        if metric in {"mean", "std", "var", "skewness"}:
            node_weights = xp.ones(river_network.n_nodes, dtype=xp.float64)
    else:
        node_weights = xp.copy(node_weights)

    if edge_weights is not None:
        edge_weights = xp.copy(edge_weights)

    func = metrics_func_finder(metric, xp).func

    weighted_field = flow(
        xp,
        river_network,
        xp.zeros(field.shape),
        func,
        invert_graph,
        node_additive_weight=field if node_weights is None else field * node_weights,
        node_modifier_use_upstream=node_modifier_use_upstream,
        edge_multiplicative_weight=edge_weights,
    )

    if metric in {"mean", "std", "var", "skewness"}:
        counts = flow(
            xp,
            river_network,
            xp.zeros(field.shape),
            func,
            invert_graph,
            node_additive_weight=xp.copy(node_weights),
            node_modifier_use_upstream=node_modifier_use_upstream,
            edge_multiplicative_weight=edge_weights,
        )

        if metric == "mean":
            weighted_field /= counts
            return weighted_field
        elif metric in {"var", "std", "skewness"}:
            weighted_sum_of_squares = flow(
                xp,
                river_network,
                xp.zeros(field.shape),
                func,
                invert_graph,
                node_additive_weight=(field**2 if node_weights is None else field**2 * node_weights),
                node_modifier_use_upstream=node_modifier_use_upstream,
                edge_multiplicative_weight=edge_weights,
            )
            mean = weighted_field / counts
            var = weighted_sum_of_squares / counts - mean**2
            var = xp.clip(var, 0, xp.inf)
            if metric == "var":
                return var
            elif metric == "std":
                return xp.sqrt(var)
            elif metric == "skewness":
                weighted_sum_of_cubes = flow(
                    xp,
                    river_network,
                    xp.zeros(field.shape),
                    func,
                    invert_graph,
                    node_additive_weight=(field**3 if node_weights is None else field**3 * node_weights),
                    node_modifier_use_upstream=node_modifier_use_upstream,
                    edge_multiplicative_weight=edge_weights,
                )
                third_moment = (
                    weighted_sum_of_cubes / counts - 3 * mean * (weighted_sum_of_squares / counts) + 2 * mean**3
                )
                return xp.where(var == 0, xp.nan, third_moment / var**1.5)
    else:
        return weighted_field
