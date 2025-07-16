import numpy as np
import xarray as xr

from earthkit.hydro.utils.convert import points_to_1d_indices, points_to_numpy
from earthkit.hydro.utils.decorators import xarray_mask
from earthkit.hydro.zonal.metrics import calculate_zonal_metric

from .find import find


def __calculate_subcatchment_metric(
    river_network,
    field,
    stations_1d,
    metric,
    node_weights=None,
    mv=np.nan,
    accept_missing=False,
):
    initial_field = np.zeros(river_network.n_nodes, dtype=int)
    unique_labels = np.arange(stations_1d.shape[0]) + 1
    initial_field[stations_1d] = unique_labels
    labels = find(river_network, initial_field)  # TODO: can skip redoing calcs here
    metric_at_stations = calculate_zonal_metric(
        field,
        labels,
        metric,
        node_weights,
        mv,
        0,  # missing labels value
        accept_missing,
    )
    return metric_at_stations[..., stations_1d]


@xarray_mask
def _calculate_subcatchment_metric(
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
    return __calculate_subcatchment_metric(
        river_network, field, points, metric, node_weights, mv, accept_missing
    )


def calculate_subcatchment_metric(
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
    Calculates the metric over the subcatchments defined by stations.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    stations : list of tuples
        List of tuple indices of the stations.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum", "prod",
        "std", "var".
    weights : ndarray
        Used to weight the field when computing the metric. Default is None.
    mv : scalar
        Missing value for the input field. Default is np.nan.
    accept_missing : bool
        Whether or not to accept missing values in the input field. Default is False.

    Returns
    -------
    dict
        Numpy array of values for each station.
    """
    assert edge_weights is None
    if isinstance(points, np.ndarray):
        stations_1d = points
    elif isinstance(points, list):
        stations_1d = points_to_numpy(points)
        stations_1d = points_to_1d_indices(river_network, stations_1d)
    elif isinstance(points, dict):
        assert isinstance(field, (xr.DataArray, xr.Dataset))
        lats = field.lat.values
        lons = field.lon.values

        indices = []
        for lat_val, lon_val in points.values():
            ilat = np.abs(lats - lat_val).argmin()
            ilon = np.abs(lons - lon_val).argmin()
            indices.append((int(ilat), int(ilon)))

        stations_1d = points_to_numpy(indices)
        stations_1d = points_to_1d_indices(river_network, stations_1d)
    else:
        raise ValueError(f"points of type {type(points)} is not supported.")

    return _calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
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
    return calculate_subcatchment_metric(
        river_network,
        field,
        points,
        "mean",
        node_weights,
        edge_weights,
        mv,
        accept_missing,
    )
