import os

import numpy as np

from earthkit.hydro.data_structures.network import RiverNetwork

from ._accumulate import _ufunc_to_downstream
from .flow import propagate


def flow_downstream(
    river_network,
    field,
    ufunc=np.add,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):
    invert_graph = False
    return flow(
        river_network,
        field,
        ufunc,
        invert_graph,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    )


def flow_upstream(
    river_network,
    field,
    ufunc=np.add,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):
    invert_graph = True
    return flow(
        river_network,
        field,
        ufunc,
        invert_graph,
        node_additive_weight,
        node_multiplicative_weight,
        node_modifier_use_upstream,
        edge_additive_weight,
        edge_multiplicative_weight,
    )


def flow(
    river_network: RiverNetwork,
    field,
    ufunc=np.add,
    invert_graph=False,
    node_additive_weight=None,
    node_multiplicative_weight=None,
    node_modifier_use_upstream=True,
    edge_additive_weight=None,
    edge_multiplicative_weight=None,
):

    use_rust = int(os.environ.get("USE_RUST", "-1"))
    if use_rust == 1 and ufunc is np.add:
        from earthkit.hydro._rust import _flow_rust

        _flow_rust(
            field,
            river_network.groups,
            invert_graph,
            node_modifier_use_upstream,
            node_additive_weight,
            node_multiplicative_weight,
            edge_additive_weight,
            edge_multiplicative_weight,
        )

        return field
    elif ufunc is np.add and use_rust != 0:
        try:
            from earthkit.hydro._rust import _flow_rust
        except (ModuleNotFoundError, ImportError):
            return flow_python(
                river_network,
                field,
                ufunc,
                invert_graph,
                node_additive_weight,
                node_multiplicative_weight,
                node_modifier_use_upstream,
                edge_additive_weight,
                edge_multiplicative_weight,
            )
        _flow_rust(
            field,
            river_network.groups,
            invert_graph,
            node_modifier_use_upstream,
            node_additive_weight,
            node_multiplicative_weight,
            edge_additive_weight,
            edge_multiplicative_weight,
        )
    else:
        return flow_python(
            river_network,
            field,
            ufunc,
            invert_graph,
            node_additive_weight,
            node_multiplicative_weight,
            node_modifier_use_upstream,
            edge_additive_weight,
            edge_multiplicative_weight,
        )

        return field


def flow_python(
    river_network,
    field,
    ufunc=np.add,
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
            ufunc=ufunc,
        )

    field = propagate(
        river_network,
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
