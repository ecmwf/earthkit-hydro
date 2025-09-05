from earthkit.hydro.length.array import _operations


def min(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_type=None,
):
    r"""
    Calculates the minimum length to all points from a set of start
    locations.

    For each node in the network, calculates the minimum length starting from any of the start locations.

    The length is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        l_j &= w_j ~\text{for start locations}\\
        l_j &= \mathrm{min}(\infty,~\mathrm{min}_{i \in \mathrm{Neighbour}(j)} l_i) + w_j
        \end{align*}

    where:

    - :math:`w_i` is the node length (e.g., pixel length),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`l_j` is the total length at node :math:`j`.

    Unreachable nodes are given a length of :math:`\infty`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    locations : array-like or dict
        A list of source nodes.
    field : array-like, optional
        An array containing length values defined on river network nodes or gridcells.
        Default is `xp.ones(river_network.n_nodes)`.
    upstream : bool, optional
        Whether or not to consider upstream lengths. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream lengths. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of minimum lengths for every river network node or gridcell, depending on `return_type`.
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
    Calculates the maximum length to all points from a set of start
    locations.

    For each node in the network, calculates the maximum length starting from any of the start locations.

    The length is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        l_j &= w_j ~\text{for start locations}\\
        l_j &= \mathrm{max}(-\infty,~\mathrm{max}_{i \in \mathrm{Neighbour}(j)} l_i) + w_j
        \end{align*}

    where:

    - :math:`w_i` is the node length (e.g., pixel length),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`l_j` is the total length at node :math:`j`.

    Unreachable nodes are given a length of :math:`-\infty`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    locations : array-like or dict
        A list of source nodes.
    field : array-like or xarray object, optional
        An array containing length values defined on river network nodes or gridcells.
        Default is `xp.ones(river_network.n_nodes)`.
    upstream : bool, optional
        Whether or not to consider upstream lengths. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream lengths. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of maximum lengths for every river network node or gridcell, depending on `return_type`.
    """
    return _operations.max(
        river_network, field, locations, upstream, downstream, return_type
    )


def to_source(
    river_network,
    field=None,
    path="shortest",
    return_type=None,
):
    r"""
    Calculates the maximum length to all points from from the river network sources.

    For each node in the network, calculates the maximum length starting from any source.

    The length is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        l_j &= w_j ~\text{for sources}\\
        l_j &= \bigoplus \left(-\infty,~\bigoplus_{i \in \mathrm{Neighbour}(j)} l_i \right) + w_j
        \end{align*}

    where:

    - :math:`w_i` is the node length (e.g., pixel length),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`\bigoplus` is the aggregation function (max for longest path or min for shortest path).
    - :math:`l_j` is the total length at node :math:`j`.

    Unreachable nodes are given a distance of :math:`-\infty` if :math:`\bigoplus` is a maximum, and :math:`\infty` if :math:`\bigoplus` is a minimum.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like, optional
        An array containing length values defined on river network nodes or gridcells.
        Default is `xp.ones(river_network.n_nodes)`.
    path : str, optional
        Whether to compute the longest or shortest path. Default is "shortest".
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of maximum lengths for every river network node or gridcell, depending on `return_type`.
    """
    locations = river_network.sources
    if path == "longest":
        return max(river_network, locations, field, True, False, return_type)
    elif path == "shortest":
        return min(river_network, locations, field, True, False, return_type)
    else:
        raise ValueError("path must be 'longest' or 'shortest'")


def to_sink(
    river_network,
    field=None,
    path="shortest",
    return_type=None,
):
    r"""
    Calculates the maximum length to all points from from the river network sinks.

    For each node in the network, calculates the maximum length starting from any sink.

    The length is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        l_j &= w_j ~\text{for sinks}\\
        l_j &= \bigoplus \left(-\infty,~\bigoplus_{i \in \mathrm{Neighbour}(j)} l_i \right) + w_j
        \end{align*}

    where:

    - :math:`w_i` is the node length (e.g., pixel length),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`\bigoplus` is the aggregation function (max for longest path or min for shortest path).
    - :math:`l_j` is the total length at node :math:`j`.

    Unreachable nodes are given a distance of :math:`-\infty` if :math:`\bigoplus` is a maximum, and :math:`\infty` if :math:`\bigoplus` is a minimum.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like, optional
        An array containing length values defined on river network nodes or gridcells.
        Default is `xp.ones(river_network.n_nodes)`.
    path : str, optional
        Whether to compute the longest or shortest path. Default is "shortest".
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    array-like
        Array of maximum lengths for every river network node or gridcell, depending on `return_type`.
    """
    locations = river_network.sinks
    if path == "longest":
        return max(river_network, locations, field, True, False, return_type)
    elif path == "shortest":
        return min(river_network, locations, field, True, False, return_type)
    else:
        raise ValueError("path must be 'longest' or 'shortest'")
