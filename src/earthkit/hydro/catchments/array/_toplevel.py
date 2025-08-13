from earthkit.hydro.catchments.array import _operations


def var(river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted variance of a field over the upstream
    catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted variance is defined as:

    .. math::

        \mathrm{var}_v = {\frac{ \sum_{i \in \mathcal{U}(v)} w_i \cdot \pi_{iv} \cdot (x_i - \mathrm{mean}_v)^2 }{ \sum_{i \in \mathcal{U}(v)} w_i \cdot \pi_{iv} }}

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`\mathrm{mean}_v` is the upstream weighted mean at node :math:`v`,
        - :math:`w_i` is the weight associated with node :math:`i`,
        - :math:`\pi_{iv}` is the product of edge weights along the path(s) from node :math:`i` to :math:`v`,
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of variance values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.var(river_network, field, locations, node_weights, edge_weights)


def std(river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted standard deviation of a field over the
    upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted standard deviation is defined as:

    .. math::

        \mathrm{std}_v = \sqrt{\frac{ \sum_{i \in \mathcal{U}(v)} w_i \cdot \pi_{iv} \cdot (x_i - \mathrm{mean}_v)^2 }{ \sum_{i \in \mathcal{U}(v)} w_i \cdot \pi_{iv} }}

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`\mathrm{mean}_v` is the upstream weighted mean at node :math:`v`,
        - :math:`w_i` is the weight associated with node :math:`i`,
        - :math:`\pi_{iv}` is the product of edge weights along the path(s) from node :math:`i` to :math:`v`,
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of standard deviation values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.std(river_network, field, locations, node_weights, edge_weights)


def mean(river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted mean of a field over the upstream catchment of
    each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted mean is defined as:

    .. math::

        \mathrm{mean}_{v} = \frac{ \sum_{i \in \mathcal{U}(v)} x_i \cdot w_i \cdot \pi_{iv} }{ \sum_{i \in \mathcal{U}(v)} w_i \cdot \pi_{iv} }

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of mean values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.mean(river_network, field, locations, node_weights, edge_weights)


def sum(river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted sum of a field over the upstream catchment of
    each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted sum is defined as:

    .. math::

        \mathrm{sum}_{v} = \sum_{i \in \mathcal{U}(v)} x_i \cdot w_i \cdot \pi_{iv}

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of sum values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.sum(river_network, field, locations, node_weights, edge_weights)


def min(river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted minimum of a field over the upstream catchment
    of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted minimum is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of minimum values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.min(river_network, field, locations, node_weights, edge_weights)


def max(xp, river_network, field, locations, node_weights=None, edge_weights=None):
    r"""
    Computes the weighted maximum of a field over the upstream catchment
    of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted maximum is defined as:

    .. math::

        \mathrm{min}_{v} = TODO: fill

    where:
        - :math:`x_i` is the value of the field at upstream node :math:`i`,
        - :math:`w_i` is the weight associated with node :math:`i` (default is 1),
        - :math:`\pi_{iv}` is the cumulative product of edge weights along all paths from node :math:`i` to node :math:`v` (default is 1),
        - :math:`\mathcal{U}(v)` is the set of all nodes upstream of :math:`v`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array, optional
        Array of weights for each node.
    edge_weights : array, optional
        Array of weights for each edge.

    Returns
    -------
    array
        Array of maximum values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.max(river_network, field, locations, node_weights, edge_weights)


def find(river_network, locations):
    r"""
    Delineates catchment boundaries.

    TODO: better description

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array containing field values defined on nodes of the river network.

    Returns
    -------
    array
        Array of labelled nodes.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return _operations.find(river_network, locations)
