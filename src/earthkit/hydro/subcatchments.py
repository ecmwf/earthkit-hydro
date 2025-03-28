from functools import partial

import numpy as np

from .catchments import _find_catchments_2D, _find_catchments_ND
from .core import flow
from .metrics import metrics_dict
from .utils import mask_2d, mask_and_unmask
from .zonal import calculate_zonal_metric


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
    stations : list of tuples
        List of tuple indices of the stations.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum", "product"
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
        labels = find(river_network, points, skip=True)
        metric_at_stations = calculate_zonal_metric(
            field.T,
            labels,
            metric,
            weights.T if weights is not None else None,
            mv,
            0,  # missing labels value
            accept_missing,
        )
        return {x: metric_at_stations[y] for (x, y) in zip(stations, labels[stations])}

    # transform here list of tuples (indices) into a tuple of lists
    # (easier to manipulate)
    stations = np.array(stations)
    stations = (stations[:, 0], stations[:, 1])

    node_numbers = np.cumsum(river_network.mask) - 1
    valid_stations = river_network.mask[stations]
    stations = tuple(station_index[valid_stations] for station_index in stations)
    stations_1d = node_numbers.reshape(river_network.mask.shape)[stations]
    points = np.zeros(river_network.n_nodes, dtype=int)
    unique_labels = np.arange(stations_1d.shape[0]) + 1
    points[stations_1d] = unique_labels
    labels = find(river_network, points, skip=True)
    metric_at_stations = calculate_zonal_metric(
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


@mask_and_unmask
def find(river_network, field, mv=0, in_place=False):
    """Labels the catchments given a field of labelled sinks.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    mv : scalar, optional
        The missing value indicator. Default is 0.
    in_place : bool, optional
        If True, modifies the input field in place. Default is False.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """
    if not in_place:
        field = field.copy()

    if len(field.shape) == 1:
        op = _find_catchments_2D
    else:
        op = _find_catchments_ND

    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, overwrite=False)

    return flow(river_network, field, True, operation, mv)


for metric in metrics_dict.keys():

    func = partial(calculate_subcatchment_metric, metric=metric)

    globals()[metric] = func
