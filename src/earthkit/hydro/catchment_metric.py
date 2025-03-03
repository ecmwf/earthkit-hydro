import numpy as np

from .accumulation import calculate_upstream_metric
from .catchment import find_subcatchments
from .label import calculate_metric_for_labels
from .utils import mask_2d


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
        return dict(zip(stations, upstream_metric_field))

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
    if isinstance(stations, np.ndarray):
        points = np.zeros(river_network.n_nodes, dtype=int)
        points[stations] = np.arange(stations.shape[0]) + 1
        labels = find_subcatchments(river_network, points, skip=True)
        return calculate_metric_for_labels(
            field.T,
            labels,
            metric,
            weights,
            mv,
            0,  # missing labels value
            accept_missing,
        )

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
        weights,
        mv,
        0,  # missing labels value
        accept_missing,
    )
    return {
        (x, y): metric_at_stations[z] for (x, y, z) in zip(*stations, unique_labels)
    }
