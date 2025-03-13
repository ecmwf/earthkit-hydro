import numpy as np

from earthkit.hydro.upstream import calculate_upstream_metric

from ..utils import mask_2d


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
    stations : tuple
        Tuple of indices of the stations.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum"
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


def mean(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_catchment_metric(
        river_network,
        field,
        stations,
        "mean",
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

    return calculate_catchment_metric(
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

    return calculate_catchment_metric(
        river_network,
        field,
        stations,
        "min",
        weights,
        mv,
        accept_missing,
    )


def sum(
    river_network,
    field,
    stations,
    weights=None,
    mv=np.nan,
    accept_missing=False,
):

    return calculate_catchment_metric(
        river_network,
        field,
        stations,
        "sum",
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

    return calculate_catchment_metric(
        river_network,
        field,
        stations,
        "product",
        weights,
        mv,
        accept_missing,
    )
