import numpy as np

from .utils import mask_and_unmask, missing_to_nan, nan_to_missing


@mask_and_unmask
def move_downstream(
    river_network,
    field,
    mv=np.nan,
    ufunc=np.add,
    accept_missing=False,
):
    """Sets each node to be the sum of its upstream nodes values, or a missing value.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field representing node values.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    ufunc : callable, optional
        The numpy ufunc to perform when propagating (default is numpy.add).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False).

    Returns
    -------
    numpy.ndarray
        The updated field with upstream contributions.

    """

    field, field_dtype = missing_to_nan(field, mv, accept_missing)

    ups = np.zeros(field.shape, dtype=field_dtype)
    mask = (
        river_network.downstream_nodes != river_network.n_nodes
    )  # remove sinks since they have no downstream
    nodes_to_update = river_network.downstream_nodes[mask]
    values_to_add = field[mask]
    ufunc.at(ups, (nodes_to_update, *[slice(None)] * (ups.ndim - 1)), values_to_add)

    ups = nan_to_missing(ups, field_dtype, mv)

    return ups


@mask_and_unmask
def move_upstream(river_network, field, mv=np.nan, accept_missing=False):
    """Sets each node to be its downstream node value, or a missing value.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field representing node values.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False).

    Returns
    -------
    numpy.ndarray
        The updated field with downstream values.

    """

    field, field_dtype = missing_to_nan(field, mv, accept_missing)

    down = np.zeros(field.shape, dtype=field_dtype)
    mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks
    down[mask] = field[river_network.downstream_nodes[mask], ...]

    down = nan_to_missing(down, field_dtype, mv)

    return down
