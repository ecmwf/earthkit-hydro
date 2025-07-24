import numpy as np
import xarray as xr
from earthkit.utils.array import array_namespace

from earthkit.hydro.upstream.array.operations import calculate_upstream_metric
from earthkit.hydro.utils.convert import points_to_1d_indices, points_to_numpy
from earthkit.hydro.utils.decorators import xarray_mask


def __calculate_catchment_metric(
    river_network,
    field,
    stations_1d,
    metric,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    upstream_metric_field = calculate_upstream_metric(
        river_network,
        field,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )
    upstream_field_at_stations = upstream_metric_field[..., stations_1d]
    return upstream_field_at_stations


@xarray_mask
def _calculate_catchment_metric(
    river_network,
    field,
    points,
    metric,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
    station_names=None,
):
    return __calculate_catchment_metric(
        river_network,
        field,
        points,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def calculate_catchment_metric(
    river_network,
    field,
    points,
    metric,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    """
    Calculates the metric over the catchments defined by the points.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    points : list of tuples
        List of tuple indices of the points.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum", "prod",
        "std", "var".
    weights : ndarray, optional
        Used to weight the field when computing the metric. Default is None.
    mv : scalar, optional
        Missing value for the input field. Default is np.nan.
    accept_missing : bool, optional
        Whether or not to accept missing values in the input field. Default is False.

    Returns
    -------
    dict
        Dictionary with (station, catchment_metric) pairs.

    """
    # TODO: Future idea could be to find all
    # nodes relevant for computing upstream
    # average, then creating a river subnetwork
    # and calculating upstream metric only there
    # (should be quicker, particularly for
    # small numbers of points)

    if isinstance(field, (xr.Dataset, xr.DataArray)):
        xp = array_namespace(field.data)
    else:
        xp = array_namespace(field)

    if isinstance(points, xp.ndarray):
        stations_1d = points
    elif isinstance(points, list):
        stations_1d = points_to_numpy(xp, points)
        stations_1d = points_to_1d_indices(river_network, xp, stations_1d)
    elif isinstance(points, dict):
        assert isinstance(field, (xr.DataArray, xr.Dataset))
        lats = field.lat.data
        lons = field.lon.data

        indices = []
        for lat_val, lon_val in points.values():
            ilat = np.abs(lats - lat_val).argmin()
            ilon = np.abs(lons - lon_val).argmin()
            indices.append((int(ilat), int(ilon)))

        stations_1d = points_to_numpy(xp, indices)
        stations_1d = points_to_1d_indices(river_network, xp, stations_1d)
    else:
        raise ValueError(f"points of type {type(points)} is not supported.")

    return _calculate_catchment_metric(
        river_network,
        field,
        stations_1d,
        metric,
        node_weights,
        edge_weights,
        mv,
        accept_missing,
        station_names=(
            list(points.keys()) if isinstance(points, dict) else np.array(points)
        ),
    )


def sum(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "sum",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def max(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "max",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def min(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "min",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def prod(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "prod",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def std(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "std",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def var(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "var",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )


def mean(
    river_network,
    field,
    points,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    return calculate_catchment_metric(
        river_network,
        field,
        points,
        "mean",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )
