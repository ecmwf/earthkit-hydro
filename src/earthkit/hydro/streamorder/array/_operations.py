from earthkit.hydro._core.flow import propagate
from earthkit.hydro._utils.decorators import mask, multi_backend
from earthkit.hydro.upstream.array._operations import sum as upstream_sum


def _ufunc_strahler(
    field,
    did,
    uid,
    eid,
    xp,
):
    maxes, counts = field
    up_maxes = xp.gather(maxes, uid, axis=-1)
    old_maxes = xp.gather(maxes, did, axis=-1)
    xp.maximum.at(maxes, did, up_maxes)
    counts[did] = (old_maxes == maxes[did]).astype(int) * counts[uid]
    xp.add.at(counts, did, (up_maxes == maxes[did]).astype(int))
    maxes[did] = maxes[did] + (counts[did] > 1).astype(int)
    return (maxes, counts)


def flow_strahler(
    xp,
    river_network,
    field,
    count,
    invert_graph=False,
):
    op = _ufunc_strahler

    def operation(
        x,
        did,
        uid,
        eid,
    ):
        return op(
            x,
            did,
            uid,
            eid,
            xp=xp,
        )

    field, count = propagate(
        river_network,
        river_network.groups,
        (field, count),
        invert_graph,
        operation,
    )

    return field


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def strahler(xp, river_network, return_type):

    field = xp.zeros(river_network.n_nodes, dtype=int)
    # TODO: make xp-agnostic
    field[river_network.sources] = 1
    counts = xp.zeros(river_network.n_nodes, dtype=int)

    decorated_func = mask(return_type == "gridded")(flow_strahler)
    return decorated_func(xp, river_network, field, counts)


@multi_backend(jax_static_args=["xp", "river_network", "return_type"])
def shreve(xp, river_network, return_type):
    field = xp.zeros(river_network.n_nodes, dtype=int)
    # TODO: make xp-agnostic
    field[river_network.sources] = 1

    return upstream_sum(
        river_network=river_network,
        field=field,
        node_weights=None,
        edge_weights=None,
        return_type=return_type,
    )
