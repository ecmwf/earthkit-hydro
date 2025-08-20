from earthkit.hydro._core.accumulate import flow_downstream, flow_upstream
from earthkit.hydro._core.metrics import metrics_func_finder


def min(xp, river_network, field, locations, upstream, downstream):

    func_obj = metrics_func_finder("min", xp)

    out = xp.full(river_network.n_nodes, func_obj.base_val)

    out[locations] = field[locations]

    func = func_obj.func

    if downstream:
        out = flow_downstream(
            xp,
            river_network,
            out,
            func,
            node_additive_weight=field,
            node_modifier_use_upstream=False,
        )
    if upstream:
        out = flow_upstream(
            xp,
            river_network,
            out,
            func,
            node_additive_weight=field,
            node_modifier_use_upstream=True,
        )

    return out


def max(xp, river_network, field, locations, upstream, downstream):

    func_obj = metrics_func_finder("min", xp)

    out = xp.full(river_network.n_nodes, func_obj.base_val)

    out[locations] = field[locations]

    func = func_obj.func

    if downstream:
        out = flow_downstream(
            xp,
            river_network,
            out,
            func,
            node_additive_weight=field,
            node_modifier_use_upstream=False,
        )
    if upstream:
        out = flow_upstream(
            xp,
            river_network,
            out,
            func,
            node_additive_weight=field,
            node_modifier_use_upstream=True,
        )

    return out
