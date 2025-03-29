import numpy as np

from .accumulation import flow_downstream, flow_upstream


def min(
    river_network, points, weights=None, upstream=False, downstream=True, mv=np.nan
):
    weights = np.ones(river_network.n_nodes) if weights is None else weights
    field = np.empty(river_network.shape)  # TODO: make 1d
    field.fill(np.inf)

    points = np.array(points)
    points = (points[:, 0], points[:, 1])
    field[points] = 0
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
