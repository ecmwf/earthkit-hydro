from functools import wraps

import numpy as np
import xarray as xr

import earthkit.hydro.distance.array._operations as array
from earthkit.hydro._backends.find import get_array_backend
from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.catchments.array._operations import preprocess_stations


def _convert_locations(func):
    @wraps(func)
    def wrapper(river_network, field, locations, *args, **kwargs):
        xp = get_array_backend(river_network.groups[0])
        orig_locations = locations
        xr_present = isinstance(field, (xr.DataArray, xr.Dataset))
        dict_locations = isinstance(locations, dict)
        if dict_locations:
            assert xr_present

            lats = field.lat.data
            lons = field.lon.data

            # TODO: decide if acceptable to always use np here
            locations = []
            for lat_val, lon_val in orig_locations.values():
                ilat = np.abs(lats - lat_val).argmin()
                ilon = np.abs(lons - lon_val).argmin()
                locations.append((int(ilat), int(ilon)))

        locations = xp.asarray(locations, device=river_network.device)
        stations_1d = preprocess_stations(xp, river_network, locations)

        result = func(river_network, field, stations_1d, *args, **kwargs)

        return result

    return wrapper


@_convert_locations
@xarray
def min(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the minimum distance to all points from a set of start locations.

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
    return array.min(river_network, field, locations, upstream, downstream)


@_convert_locations
@xarray
def max(river_network, field, locations, upstream=False, downstream=True):
    r"""
    Calculates the maximum distance to all points from a set of start locations.

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
