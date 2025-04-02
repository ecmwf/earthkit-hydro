import numpy as np

from .accumulation import flow_downstream, flow_upstream
from .utils import mask_2d, points_to_1d_indices, points_to_numpy


@mask_2d
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
    if upstream and downstream:
        # TODO: define how this should work
        # can one overwrite a starting station's distance?
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
