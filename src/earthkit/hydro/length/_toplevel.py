import earthkit.hydro.length.array._operations as array
from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.distance._toplevel import _convert_locations


@_convert_locations
@xarray
def min(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the minimum length to all points from a set of start locations.

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
    field : array-like or xarray object
        An array containing length values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether or not to consider upstream lengths.
    downstream : bool, optional
        Whether or not to consider downstream lengths.

    Returns
    -------
    array-like or xarray object
        Array of minimum lengths for every node in the river network.
    """
    return array.min(river_network, field, locations, upstream, downstream)


@_convert_locations
@xarray
def max(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the maximum length to all points from a set of start locations.

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
    field : array-like or xarray object
        An array containing length values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether or not to consider upstream lengths.
    downstream : bool, optional
        Whether or not to consider downstream lengths.

    Returns
    -------
    array-like or xarray object
        Array of maximum lengths for every node in the river network.
    """
    return array.max(river_network, field, locations, upstream, downstream)


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
