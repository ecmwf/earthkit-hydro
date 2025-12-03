from earthkit.hydro._core.accumulate import flow_downstream, flow_upstream
from earthkit.hydro._core.metrics import metrics_func_finder


def min(xp, river_network, field, locations, upstream, downstream):

    func_obj = metrics_func_finder("min", xp)

    out = xp.full(river_network.n_nodes, func_obj.base_val)

    out[locations] = 0

    func = func_obj.func

    # make xp-agnostic
    mask = xp.full(river_network.n_nodes, False)
    mask[river_network.sinks] = True
    field = field[~mask]

    if downstream:
        out = flow_downstream(xp, river_network, out, func, edge_additive_weight=field)

    if upstream:
        out = flow_upstream(xp, river_network, out, func, edge_additive_weight=field)

    return out


def max(xp, river_network, field, locations, upstream, downstream):

    func_obj = metrics_func_finder("max", xp)

    out = xp.full(river_network.n_nodes, func_obj.base_val)

    out[locations] = 0

    func = func_obj.func

    # make xp-agnostic
    mask = xp.full(river_network.n_nodes, False)
    mask[river_network.sinks] = True
    field = field[~mask]

    if downstream:
        out = flow_downstream(xp, river_network, out, func, edge_additive_weight=field)
    if upstream:
        out = flow_upstream(xp, river_network, out, func, edge_additive_weight=field)

    return out
