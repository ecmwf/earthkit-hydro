import numpy as np

from .accumulation import flow_downstream, flow_upstream
from .utils import mask_2d, points_to_1d_indices, points_to_numpy


@mask_2d
def min(
    river_network, points, weights=None, upstream=False, downstream=True, mv=np.nan
):
    """
    Calculate the minimum length to a set of points in a river network.
    The length is calculated along the river network, and can be
    computed in both/either upstream and downstream directions.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    points : list of tuples
        List of tuple indices of the points.
    weights : numpy.ndarray, optional
        length to the downstream point. Default is None, which
        corresponds to a unit length for all points.
    upstream : bool, optional
        If True, calculates the length in the upstream direction.
        Default is False.
    downstream : bool, optional
        If True, calculate the length in the downstream direction.
        Default is True.
    mv : scalar, optional
        The missing value indicator. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The length to the points in the river network.
    """

    if weights is None:
        weights = np.ones(river_network.n_nodes)
    else:
        # maybe check sinks are all zero or nan length
        pass
    field = np.empty(river_network.n_nodes)
    field.fill(np.inf)

    if isinstance(points, np.ndarray):
        points_1d = points
    else:
        points = points_to_numpy(points)
        points_1d = points_to_1d_indices(river_network, points)

    field[points_1d] = weights[points_1d]

    if downstream:
        field = flow_downstream(
            river_network,
            field,
            mv,
            ufunc=np.minimum,
            additive_weight=weights,
            modifier_use_upstream=False,
        )
    if upstream:
        field = flow_upstream(
            river_network,
            field,
            mv,
            ufunc=np.minimum,
            additive_weight=weights,
            modifier_use_upstream=True,
        )

    out_field = np.empty(river_network.shape, dtype=field.dtype)
    out_field[..., river_network.mask] = field
    out_field[..., ~river_network.mask] = mv

    return out_field


@mask_2d
def max(
    river_network, points, weights=None, upstream=False, downstream=True, mv=np.nan
):
    """
    Calculate the maximum length to a set of points in a river network.
    The length is calculated along the river network, and can be
    computed in both/either upstream and downstream directions.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    points : list of tuples
        List of tuple indices of the points.
    weights : numpy.ndarray, optional
        length to the downstream point. Default is None, which
        corresponds to a unit length for all points.
    upstream : bool, optional
        If True, calculates the length in the upstream direction.
        Default is False.
    downstream : bool, optional
        If True, calculate the length in the downstream direction.
        Default is True.
    mv : scalar, optional
        The missing value indicator. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The length to the points in the river network.
    """

    if upstream and downstream:
        # TODO: define how this should work
        # can one overwrite a starting station's length?
        #
        # NB: there is no nice way to do this as downstream
        # and then upstream like for min
        # because we would need to know the paths
        # to avoid looping over each other
        # Only way I think can think is doing each station
        # separately, but this will be very slow...
        raise NotImplementedError(
            "Max length both upstream and downstream is not yet implemented."
        )

    weights = np.ones(river_network.n_nodes) if weights is None else weights

    field = np.empty(river_network.n_nodes)
    field.fill(-np.inf)

    if isinstance(points, np.ndarray):
        points_1d = points
    else:
        points = points_to_numpy(points)
        points_1d = points_to_1d_indices(river_network, points)

    field[points_1d] = weights[points_1d]

    if downstream:
        field = flow_downstream(
            river_network,
            field,
            mv,
            ufunc=np.maximum,
            additive_weight=weights,
            modifier_use_upstream=False,
        )
    if upstream:
        field = flow_upstream(
            river_network,
            field,
            mv,
            ufunc=np.maximum,
            additive_weight=weights,
            modifier_use_upstream=True,
        )

    field = np.nan_to_num(field, neginf=np.inf)

    out_field = np.empty(river_network.shape, dtype=field.dtype)
    out_field[..., river_network.mask] = field
    out_field[..., ~river_network.mask] = mv

    return out_field


def to_sink(river_network, weights=None, path="shortest", mv=np.nan):
    """
    Calculate the minimum or maximum length to the sinks of a river network.
    The length is calculated along the river network.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    weights : numpy.ndarray, optional
        length to the downstream point. Default is None, which
        corresponds to a unit length for all points.
    path : str, optional
        Whether to find the length of the shortest or longest path.
        Default is 'shortest'.
    mv : scalar, optional
        The missing value indicator. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The length to the points in the river network.
    """

    if path == "shortest":
        return min(
            river_network,
            river_network.sinks,
            weights,
            upstream=True,
            downstream=False,
            mv=mv,
        )
    elif path == "longest":
        return max(
            river_network,
            river_network.sinks,
            weights,
            upstream=True,
            downstream=False,
            mv=mv,
        )
    else:
        raise ValueError("Path must be one of 'shortest' or 'longest'.")


def to_source(river_network, weights=None, path="shortest", mv=np.nan):
    """
    Calculate the minimum or maximum length to the sources of a river network.
    The length is calculated along the river network.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    weights : numpy.ndarray, optional
        length to the downstream point. Default is None, which
        corresponds to a unit length for all points.
    path : str, optional
        Whether to find the length of the shortest or longest path.
        Default is 'shortest'.
    mv : scalar, optional
        The missing value indicator. Default is np.nan.

    Returns
    -------
    numpy.ndarray
        The length to the points in the river network.
    """

    if path == "shortest":
        return min(
            river_network,
            river_network.sources,
            weights,
            upstream=False,
            downstream=True,
            mv=mv,
        )
    elif path == "longest":
        return max(
            river_network,
            river_network.sources,
            weights,
            upstream=False,
            downstream=True,
            mv=mv,
        )
    else:
        raise ValueError("Path must be one of 'shortest' or 'longest'.")
