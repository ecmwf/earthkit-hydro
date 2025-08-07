from earthkit.hydro.downstream.array import _operations


def var(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted variance of a field over all downstream nodes.

    TODO: add better description

    The weighted variance is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of variance values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.var(
        river_network,
        field,
        node_weights,
        edge_weights,
    )


def std(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted standard deviation of a field over all
    downstream nodes.

    TODO: add better description

    The weighted standard deviation is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of standard deviation values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.std(
        river_network,
        field,
        node_weights,
        edge_weights,
    )


def mean(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted mean of a field over all downstream nodes.

    TODO: add better description

    The weighted mean is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of mean values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.mean(
        river_network,
        field,
        node_weights,
        edge_weights,
    )


def sum(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted sum of a field over all downstream nodes.

    TODO: add better description

    The weighted sum is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of sum values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.sum(
        river_network,
        field,
        node_weights,
        edge_weights,
    )


def min(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted minimum of a field over all downstream nodes.

    TODO: add better description

    The weighted minimum is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of minimum values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.min(
        river_network,
        field,
        node_weights,
        edge_weights,
    )


def max(
    river_network,
    field,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted maximum of a field over all downstream nodes.

    TODO: add better description

    The weighted maximum is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{D}(v)` is the set of all nodes downstream of :math:`v`.

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
        Array of maximum values.

    Notes
    -----
    - The function includes the location itself in its downstream set :math:`\mathcal{D}(v)`.
    """
    return _operations.max(
        river_network,
        field,
        node_weights,
        edge_weights,
    )
