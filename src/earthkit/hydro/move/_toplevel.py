from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.move import array


@xarray
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
    return array.upstream(river_network, field, node_weights, edge_weights, metric)


@xarray
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
    return array.downstream(river_network, field, node_weights, edge_weights, metric)
