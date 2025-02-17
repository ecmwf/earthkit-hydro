import numpy as np

from .utils import check_missing, is_missing, mask_and_unmask_data


@mask_and_unmask_data
def move_downstream(
    river_network, field, mv=np.nan, ufunc=np.add, accept_missing=False
):
    """Sets each node to be the sum of its upstream nodes values, or a missing value.

    Parameters
    ----------
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
    missing_values_present = check_missing(field, mv, accept_missing)

    ups = np.zeros(river_network.n_nodes, dtype=field.dtype)
    mask = (
        river_network.downstream_nodes != river_network.n_nodes
    )  # remove sinks since they have no downstream
    nodes_to_update = river_network.downstream_nodes[mask]
    values_to_add = field[mask]
    ufunc.at(ups, nodes_to_update, values_to_add)
    if missing_values_present and not np.isnan(mv):
        missing_indices = is_missing(values_to_add, mv)
        if len(field.shape) == 1:
            ups[nodes_to_update[missing_indices]] = mv
        else:
            missing_indices = np.array(np.where(missing_indices))
            missing_indices[0] = nodes_to_update[missing_indices[0]]
            ups[tuple(missing_indices)] = mv
    return ups


@mask_and_unmask_data
def move_upstream(river_network, field, mv=np.nan, accept_missing=False):
    """Sets each node to be its downstream node value, or a missing value.

    Parameters
    ----------
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
    _ = check_missing(field, mv, accept_missing)

    down = np.zeros(river_network.n_nodes, dtype=field.dtype)
    mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks
    down[mask] = field[river_network.downstream_nodes[mask]]
    return down
