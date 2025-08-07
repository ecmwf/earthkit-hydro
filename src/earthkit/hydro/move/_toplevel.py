from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.move import array


@xarray
def upstream(river_network, field, node_weights=None, edge_weights=None, metric="sum"):
    r"""
    Moves a field upstream.

    Computes a one-step neighbor accumulation (local aggregation) moving upstream only.

    The accumulation is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        n_j &= \bigoplus_{i \in \mathrm{Down}(j)} w_{ij} \cdot x'_i
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Down}(j)` is the set of downstream nodes flowing out of node :math:`j`,
    - :math:`\bigoplus` is the aggregation function (e.g. a summation).
    - :math:`n_j` is the weighted aggregated value at node :math:`j`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.
    metric : str, optional
        Aggregation function to apply.


    Returns
    -------
    array-like or xarray object
        Field after movement up the river network.
    """
    return array.upstream(river_network, field, node_weights, edge_weights, metric)


@xarray
def downstream(
    river_network, field, node_weights=None, edge_weights=None, metric="sum"
):
    r"""
    Moves a field downstream.

    Computes a one-step neighbor accumulation (local aggregation) moving downstream only.

    The accumulation is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        n_j &= \bigoplus_{i \in \mathrm{Up}(j)} w_{ij} \cdot x'_i
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`\bigoplus` is the aggregation function (e.g. a summation).
    - :math:`n_j` is the weighted aggregated value at node :math:`j`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.
    metric : str, optional
        Aggregation function to apply.


    Returns
    -------
    array-like or xarray object
        Field after movement down the river network.
    """
    return array.downstream(river_network, field, node_weights, edge_weights, metric)
