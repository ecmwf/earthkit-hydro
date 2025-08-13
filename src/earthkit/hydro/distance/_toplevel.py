import earthkit.hydro.distance.array as array
from earthkit.hydro._utils.decorators import xarray


@xarray
def min(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_grid=True,
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
    field : array-like or xarray object
        An array containing distance values defined on edges of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether or not to consider upstream distances.
    downstream : bool, optional
        Whether or not to consider downstream distances.

    Returns
    -------
    array-like or xarray object
        Array of minimum distances for every node in the river network.
    """
    return array.min(river_network, locations, field, upstream, downstream, return_grid)


@xarray
def max(
    river_network,
    locations,
    field=None,
    upstream=False,
    downstream=True,
    return_grid=True,
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
    field : array-like or xarray object
        An array containing distance values defined on edges of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether or not to consider upstream distances.
    downstream : bool, optional
        Whether or not to consider downstream distances.

    Returns
    -------
    array-like or xarray object
        Array of maximum distances for every node in the river network.
    """
    return array.max(river_network, locations, field, upstream, downstream, return_grid)


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
