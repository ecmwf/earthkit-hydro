from functools import wraps

import numpy as np
import xarray as xr

import earthkit.hydro.catchments.array._operations as array
from earthkit.hydro._backends.find import get_array_backend
from earthkit.hydro._utils.decorators import xarray
from earthkit.hydro.catchments.array._operations import preprocess_stations


def _name_last_dim(func):
    @wraps(func)
    def wrapper(river_network, field, locations, *args, **kwargs):
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

        xp = get_array_backend(river_network.groups[0])
        locations = xp.asarray(locations, device=river_network.device)
        stations_1d = preprocess_stations(xp, river_network, locations)

        if kwargs.get("output_core_dims", None) is None:
            kwargs["output_core_dims"] = [["station_index"]]
        if kwargs.get("output_sizes", None) is None:
            kwargs["output_sizes"] = {"station_index": locations.shape[0]}

        result = func(river_network, field, stations_1d, *args, **kwargs)

        if xr_present:
            if dict_locations:
                names = xp.asarray(
                    list(orig_locations.keys()), device=river_network.device
                )
                coords = xp.asarray(
                    list(orig_locations.values()), device=river_network.device
                )
                result = result.assign_coords(
                    station_name=("station_index", names),
                    lat=("station_index", coords[:, 0]),
                    lon=("station_index", coords[:, 1]),
                )

            if locations.ndim == 1:
                result = result.assign_coords(idx=("station_index", locations))
            elif locations.ndim == 2:
                # TODO: check if xp agnostic
                result = result.assign_coords(
                    xidx=("station_index", locations[:, 0]),
                    yidx=("station_index", locations[:, 1]),
                )
        return result

    return wrapper


@_name_last_dim
@xarray
def var(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted variance of a field over the upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted variance is defined as:

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
        Array of variance values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return array.var(river_network, field, locations, node_weights, edge_weights)


@_name_last_dim
@xarray
def std(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted standard deviation of a field over the upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted standard deviation is defined as:

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
        Array of standard deviation values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return array.std(river_network, field, locations, node_weights, edge_weights)


@_name_last_dim
@xarray
def mean(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted mean of a field over the upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted mean is defined as:

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
        Array of mean values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return array.mean(river_network, field, locations, node_weights, edge_weights)


@_name_last_dim
@xarray
def sum(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted sum of a field over the upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted sum is defined as:

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
        Array of sum values for each location in `locations`.

    Notes
    -----
    - The function includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    return array.sum(river_network, field, locations, node_weights, edge_weights)


@_name_last_dim
@xarray
def min(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted minimum of a field over the upstream catchment of each specified location.

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
    return array.min(river_network, field, locations, node_weights, edge_weights)


@_name_last_dim
@xarray
def max(
    river_network,
    field,
    locations,
    node_weights=None,
    edge_weights=None,
):
    r"""
    Computes the weighted maximum of a field over the upstream catchment of each specified location.

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
    return array.max(river_network, field, locations, node_weights, edge_weights)


@xarray
def find(river_network, field):
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
    - The functi on includes the location itself in its upstream set :math:`\mathcal{U}(v)`.
    """
    array.find(river_network, field)
