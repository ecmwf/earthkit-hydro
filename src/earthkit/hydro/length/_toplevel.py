import earthkit.hydro.length.array._operations as array
from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.distance._toplevel import _convert_locations


@_convert_locations
@xarray
def min(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the minimum length to all points from a set of start locations.

    TODO: improve description, and use node weights

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array of river network lengths defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether to compute distances downstream.
    downstream : bool, optional
        Whether to compute distances upstream.


    Returns
    -------
    array
        Array of lengths.
    """
    return array.min(river_network, field, locations, upstream, downstream)


@_convert_locations
@xarray
def max(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the maximum length to all points from a set of start locations.

    TODO: improve description, and use node weights

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array
        An array of river network lengths defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    upstream : bool, optional
        Whether to compute distances downstream.
    downstream : bool, optional
        Whether to compute distances upstream.


    Returns
    -------
    array
        Array of lengths.
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
