from ._accumulate import _ufunc_to_downstream
from .flow import propagate


def move_downstream(
    xp,
    river_network,
    field,
    func,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):
    invert_graph = False
    return move_python(
        xp,
        river_network,
        field,
        func,
        invert_graph,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    )


def move_upstream(
    xp,
    river_network,
    field,
    func,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):
    invert_graph = True
    return move_python(
        xp,
        river_network,
        field,
        func,
        invert_graph,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    )


def move_python(
    xp,
    river_network,
    field,
    func,
    invert_graph=False,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):
    op = _ufunc_to_downstream

    def operation(
        field,
        did,
        uid,
        eid,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    ):
        return op(
            field,
            did,
            uid,
            eid,
            node_additive_weight,
            node_multiplicative_weight,
            node_modifier_use_upstream,
            edge_additive_weight,
            edge_multiplicative_weight,
            func=func,
            xp=xp,
        )

    field = propagate(
        river_network,
        river_network.data,
        field,
        invert_graph,
        operation,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    )

    return field
