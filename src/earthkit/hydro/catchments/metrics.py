import numpy as np

from earthkit.hydro.upstream.operations import calculate_upstream_metric
from earthkit.hydro.utils.convert import points_to_1d_indices, points_to_numpy
from earthkit.hydro.utils.decorators import mask_2d


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

    if isinstance(points, np.ndarray):
        upstream_metric_field = calculate_upstream_metric(
            river_network,
            field,
            metric,
            node_weights,
            edge_weights,
            mv,
            accept_missing,
        )
        upstream_field_at_stations = upstream_metric_field[..., points]
        upstream_field_at_stations = np.moveaxis(upstream_field_at_stations, -1, 0)
        return dict(zip(points, upstream_field_at_stations))
    elif isinstance(points, list):
        points = points_to_numpy(points)

        stations_1d = points_to_1d_indices(river_network, points)

        upstream_metric_field = calculate_upstream_metric(
            river_network,
            field,
            metric,
            node_weights,
            edge_weights,
            mv,
            accept_missing,
        )

        metric_at_stations = upstream_metric_field[..., stations_1d]

        return {
            (x, y): metric_at_stations[..., i] for i, (x, y) in enumerate(zip(*points))
        }
    elif isinstance(points, dict):
        raise NotImplementedError(f"points of type {type(points)} not yet implemented.")
    else:
        raise ValueError(f"points of type {type(points)} is not supported.")


@mask_2d
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


@mask_2d
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


@mask_2d
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


@mask_2d
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


@mask_2d
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


@mask_2d
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


@mask_2d
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
