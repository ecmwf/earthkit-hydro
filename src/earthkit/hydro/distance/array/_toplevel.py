from earthkit.hydro.distance.array import _operations


def min(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_type=None,
):
    r"""
    Calculates the minimum distance to all points from a set of start
    locations.

    For each node in the network, calculates the minimum distance starting from any of the start locations.

    The distance is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        d_j &= 0 ~\text{for start locations}\\
        d_j &= \mathrm{min}(\infty,~\mathrm{min}_{i \in \mathrm{Neighbour}(j)} (d_i + w_{ij}) )
        \end{align*}

    where:

    - :math:`w_{ij}` is the edge distance (e.g., downstream distance),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`d_j` is the total distance at node :math:`j`.

    Unreachable nodes are given a distance of :math:`\infty`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    locations : array-like or dict
        A list of source nodes.
    field : array-like, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    upstream : bool, optional
        Whether or not to consider upstream distances. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream distances. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of minimum distances for every river network node or gridcell, depending on `return_type`.
    """
    return _operations.min(
        river_network, field, locations, upstream, downstream, return_type
    )


def max(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_type=None,
):
    r"""
    Calculates the maximum distance to all points from a set of start
    locations.

    For each node in the network, calculates the maximum distance starting from any of the start locations.

    The distance is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        d_j &= 0 ~\text{for start locations}\\
        d_j &= \mathrm{max}(-\infty,~\mathrm{max}_{i \in \mathrm{Neighbour}(j)} (d_i + w_{ij}) )
        \end{align*}

    where:

    - :math:`w_{ij}` is the edge distance (e.g., downstream distance),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`d_j` is the total distance at node :math:`j`.

    Unreachable nodes are given a distance of :math:`-\infty`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    locations : array-like or dict
        A list of source nodes.
    field : array-like, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    upstream : bool, optional
        Whether or not to consider upstream distances. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream distances. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of maximum distances for every river network node or gridcell, depending on `return_type`.
    """
    return _operations.max(
        river_network, field, locations, upstream, downstream, return_type
    )


def to_source(*args, **kwargs):
    r"""
    TODO: implement
    """
    raise NotImplementedError


def to_sink(*args, **kwargs):
    r"""
    TODO: implement
    """
    raise NotImplementedError
