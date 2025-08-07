from earthkit.hydro.move.array import _operations


def upstream(river_network, field, node_weights=None, edge_weights=None, metric="sum"):
    r"""
    Moves a field up a river network one step.

    TODO: improve description

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.


    Returns
    -------
    array
        Field after movement up the river network.
    """
    return _operations.upstream(
        river_network, field, node_weights, edge_weights, metric
    )


def downstream(
    river_network, field, node_weights=None, edge_weights=None, metric="sum"
):
    r"""
    Moves a field down a river network one step.

    TODO: improve description

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.


    Returns
    -------
    array
        Field after movement down the river network.
    """
    return _operations.downstream(
        river_network, field, node_weights, edge_weights, metric
    )
