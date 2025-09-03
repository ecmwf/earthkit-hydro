from earthkit.hydro.move.array import _operations


def upstream(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    metric="sum",
    return_type=None,
):
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

    Sinks are given a value of 0.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like, optional
        Array of weights for each edge. Default is None (unweighted).
    metric : str, optional
        Aggregation function to apply. Options are 'var', 'std', 'mean', 'sum', 'min' and 'max'. Default is `'sum'`.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.


    Returns
    -------
    array-like
        Array of values after movement up the river network for every river network node or gridcell, depending on `return_type`.
    """
    return _operations.upstream(
        river_network, field, node_weights, edge_weights, metric, return_type
    )


def downstream(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
    metric="sum",
    return_type=None,
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

    Sources are given a value of 0.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like
        An array containing field values defined on river network nodes or gridcells.
    node_weights : array-like, optional
        Array of weights for each river network node or gridcell. Default is None (unweighted).
    edge_weights : array-like, optional
        Array of weights for each edge. Default is None (unweighted).
    metric : str, optional
        Aggregation function to apply. Options are 'var', 'std', 'mean', 'sum', 'min' and 'max'. Default is `'sum'`.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.


    Returns
    -------
    array-like
        Array of values after movement down the river network for every river network node or gridcell, depending on `return_type`.
    """
    return _operations.downstream(
        river_network, field, node_weights, edge_weights, metric, return_type
    )
