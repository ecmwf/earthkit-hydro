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
    return_grid : bool, optional
        If True (default), return results on the full grid with nans at missing gridcells.
        If False, return a 1D array with values only on the river network graph.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of minimum distances for every river network node or gridcell, depending on `return_grid`.
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
    return_grid : bool, optional
        If True (default), return results on the full grid with nans at missing gridcells.
        If False, return a 1D array with values only on the river network graph.
    input_core_dims : sequence of sequence, optional
        List of core dimensions on each input xarray argument that should not be broadcast.
        Default is None, which attempts to autodetect input_core_dims from the xarray inputs.
        Ignored if no xarray inputs passed.

    Returns
    -------
    xarray object
        Array of maximum distances for every river network node or gridcell, depending on `return_grid`.
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
