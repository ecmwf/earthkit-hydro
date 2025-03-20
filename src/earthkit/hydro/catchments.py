from functools import partial

import numpy as np

from .core import flow
from .metrics import metrics_dict
from .upstream import calculate_upstream_metric
from .utils import is_missing, mask_2d, mask_and_unmask


@mask_2d
def calculate_catchment_metric(
    river_network,
    field,
    stations,
    metric,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):
    """
    Calculates the metric over the catchments defined by stations.

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
    # small numbers of stations)

    if isinstance(stations, np.ndarray):
        upstream_metric_field = calculate_upstream_metric(
            river_network,
            field,
            metric,
            weights,
            mv,
            False,  # not in_place!
            accept_missing,
            skip=True,
        )

        upstream_metric_field = np.transpose(
            upstream_metric_field,
            axes=[0] + list(range(upstream_metric_field.ndim - 1, 0, -1)),
        )

        return dict(zip(stations, upstream_metric_field[stations]))

    # transform here list of tuples (indices) into a tuple of lists
    # (easier to manipulate)
    stations = np.array(stations)
    stations = (stations[:, 0], stations[:, 1])

    node_numbers = np.cumsum(river_network.mask) - 1
    valid_stations = river_network.mask[stations]
    stations = tuple(station_index[valid_stations] for station_index in stations)
    stations_1d = node_numbers.reshape(river_network.mask.shape)[stations]

    upstream_metric_field = calculate_upstream_metric(
        river_network,
        field,
        metric,
        weights,
        mv,
        False,  # not in_place!
        accept_missing,
        skip=True,
    )
    metric_at_stations = upstream_metric_field[stations_1d]

    return {(x, y): metric_at_stations[i].T for i, (x, y) in enumerate(zip(*stations))}


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
        return op(river_network, field, grouping, mv, overwrite=True)

    return flow(river_network, field, True, operation, mv)


for metric in metrics_dict.keys():

    func = partial(calculate_catchment_metric, metric=metric)

    globals()[metric] = func


def _find_catchments_2D(river_network, field, grouping, mv, overwrite):
    """Updates field in-place with the value of its downstream nodes, dealing
    with missing values for 2D fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    mv : scalar
        The missing value indicator.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None

    """
    valid_group = grouping[
        ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    ]  # only update nodes where the downstream belongs to a catchment
    if not overwrite:
        valid_group = valid_group[is_missing(field[valid_group], mv)]
    field[valid_group] = field[river_network.downstream_nodes[valid_group]]


def _find_catchments_ND(river_network, field, grouping, mv, overwrite):
    """Updates field in-place with the value of its downstream nodes, dealing
    with missing values for ND fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    mv : scalar
        The missing value indicator.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None

    """
    valid_mask = ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    valid_indices = np.array(np.where(valid_mask))
    valid_indices[0] = grouping[valid_indices[0]]
    if not overwrite:
        temp_valid_indices = valid_indices[0]
        valid_mask = is_missing(field[tuple(valid_indices)], mv)
        valid_indices = np.array(np.where(valid_mask))
        valid_indices[0] = temp_valid_indices[valid_indices[0]]
    downstream_valid_indices = valid_indices.copy()
    downstream_valid_indices[0] = river_network.downstream_nodes[
        downstream_valid_indices[0]
    ]
    field[tuple(valid_indices)] = field[tuple(downstream_valid_indices)]
