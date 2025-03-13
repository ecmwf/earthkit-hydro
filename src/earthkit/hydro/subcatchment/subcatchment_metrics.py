import numpy as np

from ..catchments import find_subcatchments
from ..label import calculate_metric_for_labels
from ..utils import mask_2d


@mask_2d
def calculate_subcatchment_metric(
    river_network,
    field,
    stations,
    metric,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):
    """
    Calculates the metric over the subcatchments defined by stations.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    stations : tuple
        Tuple of indices of the stations.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum"
    weights : ndarray
        Used to weight the field when computing the metric. Default is None.
    mv : scalar
        Missing value for the input field. Default is np.nan.
    accept_missing : bool
        Whether or not to accept missing values in the input field. Default is False.

    Returns
    -------
    dict
        Dictionary with (station, catchment_metric) pairs.
    """
    if isinstance(stations, np.ndarray):
        points = np.zeros(river_network.n_nodes, dtype=int)
        points[stations] = np.arange(stations.shape[0]) + 1
        labels = find_subcatchments(river_network, points, skip=True)
        metric_at_stations = calculate_metric_for_labels(
            field.T,
            labels,
            metric,
            weights.T if weights is not None else None,
            mv,
            0,  # missing labels value
            accept_missing,
        )
        return {x: metric_at_stations[y] for (x, y) in zip(stations, labels[stations])}

    node_numbers = np.cumsum(river_network.mask) - 1
    valid_stations = river_network.mask[stations]
    stations = tuple(station_index[valid_stations] for station_index in stations)
    stations_1d = node_numbers.reshape(river_network.mask.shape)[stations]
    points = np.zeros(river_network.n_nodes, dtype=int)
    unique_labels = np.arange(stations_1d.shape[0]) + 1
    points[stations_1d] = unique_labels
    labels = find_subcatchments(river_network, points, skip=True)
    metric_at_stations = calculate_metric_for_labels(
        field.T,
        labels,
        metric,
        weights.T if weights is not None else None,
        mv,
        0,  # missing labels value
        accept_missing,
    )
    return {
        (x, y): metric_at_stations[z] for (x, y, z) in zip(*stations, unique_labels)
    }


def sum(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_subcatchment_metric(
        river_network,
        field,
        stations,
        "sum",
        weights,
        mv,
        accept_missing,
    )


def max(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_subcatchment_metric(
        river_network,
        field,
        stations,
        "max",
        weights,
        mv,
        accept_missing,
    )


def min(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_subcatchment_metric(
        river_network,
        field,
        stations,
        "min",
        weights,
        mv,
        accept_missing,
    )


def mean(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_subcatchment_metric(
        river_network,
        field,
        stations,
        "mean",
        weights,
        mv,
        accept_missing,
    )


def product(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_subcatchment_metric(
        river_network,
        field,
        stations,
        "product",
        weights,
        mv,
        accept_missing,
    )
