import numpy as np

from .accumulation import flow_downstream, flow_upstream
from .utils import points_to_1d_indices, points_to_numpy


def min(
    river_network, points, weights=None, upstream=False, downstream=True, mv=np.nan
):
    if weights is None:
        weights = np.ones(river_network.n_nodes)
    else:
        # maybe check sinks are all zero or nan distance
        pass
    field = np.empty(river_network.n_nodes)
    field.fill(np.inf)

    points = points_to_numpy(points)

    points_1d = points_to_1d_indices(points)

    field[points_1d] = 0
    if downstream:
        field = flow_downstream(
            river_network, field, mv, ufunc=np.minimum, additive_weight=weights
        )
    if upstream:
        field = flow_upstream(
            river_network, field, mv, ufunc=np.minimum, additive_weight=weights
        )
    return field


def max(
    river_network, points, weights=None, upstream=False, downstream=True, mv=np.nan
):
    if upstream and downstream:
        # TODO: define how this should work
        # can one overwrite a startiing station's distance?
        raise NotImplementedError(
            "Max distance both upstream and downstream is not yet implemented."
        )

    weights = np.ones(river_network.n_nodes) if weights is None else weights
    field = np.empty(river_network.shape)
    field.fill(-np.inf)
    points = np.array(points)
    points = (points[:, 0], points[:, 1])
    field[points] = 0
    if downstream:
        field = flow_downstream(
            river_network, field, mv, ufunc=np.maximum, additive_weight=weights
        )
    if upstream:
        field = flow_upstream(
            river_network, field, mv, ufunc=np.maximum, additive_weight=weights
        )
    return np.nan_to_num(field, neginf=np.inf)
