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
    Computes the weighted variance of a field over the upstream
    catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted variance is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       q'_i &= w'_i \cdot x_i^2 \\
       n_j &= x'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot n_i \\
       q_j &= q'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot q_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j} \\
       \mathrm{Var}(x)_j &= \frac{q_j}{d_j} - \bar{x}_j^2
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g., discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`q_j` is the accumulated weighted squared value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted average at node :math:`j`,
    - :math:`\mathrm{Var}(x)_j` is the weighted variance at node :math:`j`.

    Accumulation proceeds in topological order from the sources to the sinks. This formulation computes the population variance.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of variance values for each location in `locations`.
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
    Computes the weighted standard deviation of a field over the
    upstream catchment of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted standard deviation is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       q'_i &= w'_i \cdot x_i^2 \\
       n_j &= x'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot n_i \\
       q_j &= q'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot q_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j} \\
       \mathrm{Var}(x)_j &= \frac{q_j}{d_j} - \bar{x}_j^2 \\
       \mathrm{Std}(x)_j &= \sqrt{\mathrm{Var}(x)_j}
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g., discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`q_j` is the accumulated weighted squared value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted average at node :math:`j`,
    - :math:`\mathrm{Var}(x)_j` is the weighted variance at node :math:`j`.
    - :math:`\mathrm{Std}(x)_j` is the weighted standard deviation at node :math:`j`.

    Accumulation proceeds in topological order from the sources to the sinks. This formulation computes the population standard deviation.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of standard deviation values for each location in `locations`.
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
    Computes the weighted mean of a field over the upstream catchment of
    each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted mean is defined as:

    .. math::
       :nowrap:

       \begin{align*}
       x'_i &= w'_i \cdot x_i \\
       n_j &= x'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot n_i \\
       d_j &= w'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot d_i \\
       \bar{x}_j &= \frac{n_j}{d_j}
       \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`n_j` is the accumulated weighted value,
    - :math:`d_j` is the accumulated weight (denominator),
    - :math:`\bar{x}_j` is the weighted mean at node :math:`j`.

    Accumulation proceeds in topological order from the sources to the sinks.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of mean values for each location in `locations`.
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
    Computes the weighted sum of a field over the upstream catchment of
    each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted sum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        n_j &= x'_j + \sum_{i \in \mathrm{Up}(j)} w_{ij} \cdot n_i
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`n_j` is the weighted sum at node :math:`j`.

    Accumulation proceeds in topological order from the sources to the sinks.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of sum values for each location in `locations`.
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
    Computes the weighted minimum of a field over the upstream catchment
    of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted minimum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        m_j &= \mathrm{min}(x'_j,~\mathrm{min}_{i \in \mathrm{Up}(j)} w_{ij} \cdot m_i)
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`m_j` is the weighted minimum at node :math:`j`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of minimum values for each location in `locations`.
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
    Computes the weighted maximum of a field over the upstream catchment
    of each specified location.

    For each location, this function identifies all upstream nodes in the river network
    and accumulates their contributions downstream, weighted by both node and edge weights.

    The weighted maximum is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        x'_i &= w'_i \cdot x_i \\
        m_j &= \mathrm{max} (x'_j,~\mathrm{max}_{i \in \mathrm{Up}(j)} w_{ij} \cdot m_i)
        \end{align*}

    where:

    - :math:`x_i` is the input value at node :math:`i` (e.g., rainfall),
    - :math:`w'_i` is the node weight (e.g., pixel area),
    - :math:`w_{ij}` is the edge weight from node :math:`i` to node :math:`j` (e.g. discharge partitioning ratio),
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`m_j` is the weighted maximum at node :math:`j`.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.
    locations : array-like or dict
        A list of node indices at which to compute.
    node_weights : array-like or xarray object, optional
        Array of weights for each node.
    edge_weights : array-like or xarray object, optional
        Array of weights for each edge.

    Returns
    -------
    array-like or xarray object
        Array of maximum values for each location in `locations`.
    """
    return array.max(river_network, field, locations, node_weights, edge_weights)


@xarray
def find(river_network, field):
    r"""
    Delineates catchment areas.

    Given a field indicating one or more start locations (e.g., outlet points or pour points),
    this function delineates the catchments upstream of each start location by grouping all cells that flow into these points.

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    field : array-like or xarray object
        An array containing field values defined on nodes of the river network.

    Returns
    -------
    array-like or xarray object
        Array of labelled nodes.
    """
    array.find(river_network, field)
