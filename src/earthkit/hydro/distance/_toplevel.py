import earthkit.hydro.distance.array as array
from earthkit.hydro._utils.decorators import xarray


@xarray
def min(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_type=None,
    input_core_dims=None,
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
    field : array-like or xarray object, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    upstream : bool, optional
        Whether or not to consider upstream distances. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream distances. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of minimum distances for every river network node or gridcell, depending on `return_type`.
    """
    return array.min(river_network, locations, field, upstream, downstream, return_type)


@xarray
def max(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_type=None,
    input_core_dims=None,
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
    field : array-like or xarray object, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    upstream : bool, optional
        Whether or not to consider upstream distances. Default is False.
    downstream : bool, optional
        Whether or not to consider downstream distances. Default is True.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of maximum distances for every river network node or gridcell, depending on `return_type`.
    """
    return array.max(river_network, locations, field, upstream, downstream, return_type)


@xarray
def to_source(
    river_network,
    field=None,
    path="shortest",
    return_type=None,
    input_core_dims=None,
):
    r"""
    Calculates the maximum distance to all points from the river network sources.

    For each node in the network, calculates the maximum distance starting from any source.

    The distance is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        d_j &= 0 ~\text{for sources}\\
        d_j &= \bigoplus \left(-\infty,~\bigoplus_{i \in \mathrm{Neighbour}(j)} (d_i + w_{ij}) \right)
        \end{align*}

    where:

    - :math:`w_{ij}` is the edge distance (e.g., downstream distance),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`\bigoplus` is the aggregation function (max for longest path or min for shortest path).
    - :math:`d_j` is the total distance at node :math:`j`.

    Unreachable nodes are given a distance of :math:`-\infty` if :math:`\bigoplus` is a maximum, and :math:`\infty` if :math:`\bigoplus` is a minimum.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    path : str, optional
        Whether to compute the longest or shortest path. Default is "shortest".
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of maximum distances for every river network node or gridcell, depending on `return_type`.
    """
    return array.to_source(river_network, field, path, return_type)


@xarray
def to_sink(
    river_network,
    field=None,
    path="shortest",
    return_type=None,
    input_core_dims=None,
):
    r"""
    Calculates the maximum distance to all points from the river network sinks.

    For each node in the network, calculates the maximum distance starting from any sink.

    The distance is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        d_j &= 0 ~\text{for sinks}\\
        d_j &= \bigoplus \left(-\infty,~\bigoplus_{i \in \mathrm{Neighbour}(j)} (d_i + w_{ij}) \right)
        \end{align*}

    where:

    - :math:`w_{ij}` is the edge distance (e.g., downstream distance),
    - :math:`\mathrm{Neighbour}(j)` is the set of neighbouring nodes to node :math:`j`, which can include upstream and/or downstream nodes depending on passed arguments.
    - :math:`\bigoplus` is the aggregation function (max for longest path or min for shortest path).
    - :math:`d_j` is the total distance at node :math:`j`.

    Unreachable nodes are given a distance of :math:`-\infty` if :math:`\bigoplus` is a maximum, and :math:`\infty` if :math:`\bigoplus` is a minimum.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object, optional
        An array containing length values defined on river network edges.
        Default is `xp.ones(river_network.n_edges)`.
    path : str, optional
        Whether to compute the longest or shortest path. Default is "shortest".
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of maximum distances for every river network node or gridcell, depending on `return_type`.
    """
    return array.to_sink(river_network, field, path, return_type)
