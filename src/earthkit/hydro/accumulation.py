import numpy as np
from .utils import mask_and_unmask_data, check_missing, is_missing


@mask_and_unmask_data
def accumulate_downstream(river_network, field, mv=np.nan, in_place=False, operation=np.add, accept_missing=False):
    """
    Accumulate a field downstream along the river network.

    Parameters
    ----------
    field : numpy.ndarray
        The input field to propagate.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    in_place : bool, optional
        If True, modifies the field in-place (default is False).
    operation : callable, optional
        The operation to perform when propagating (default is numpy.add).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False).

    Returns
    -------
    numpy.ndarray
        The propagated field.
    """

    missing_values_present = check_missing(field, mv, accept_missing)

    if not in_place:
        field = field.copy()

    if not missing_values_present:
        for grouping in river_network.topological_groups[:-1]:
            operation.at(field, river_network.downstream_nodes[grouping], field[grouping])
    else:
        for grouping in river_network.topological_groups[:-1]:
            nodes_to_update = river_network.downstream_nodes[grouping]
            values_to_add = field[grouping]
            missing_indices = np.logical_or(is_missing(field[grouping], mv), is_missing(field[nodes_to_update], mv))
            operation.at(field, nodes_to_update, values_to_add)
            field[nodes_to_update[missing_indices]] = mv
    return field


@mask_and_unmask_data
def move_downstream(river_network, field, mv=np.nan, operation=np.add, accept_missing=False):
    """
    Sets each node to be the sum of its upstream nodes values, or a missing value.

    Parameters
    ----------
    field : numpy.ndarray
        The input field representing node values.
    mv : scalar, optional
        The missing value to use (default is np.nan).
    operation : callable, optional
        The operation to perform when propagating (default is numpy.add).
    accept_missing : bool, optional
        If True, missing values are allowed in the input field (default is False

    Returns
    -------
    numpy.ndarray
        The updated field with upstream contributions.
    """
    missing_values_present = check_missing(field, mv, accept_missing)

    ups = np.zeros(river_network.n_nodes, dtype=field.dtype)
    mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks since they have no downstream
    nodes_to_update = river_network.downstream_nodes[mask]
    values_to_add = field[mask]
    operation.at(ups, nodes_to_update, values_to_add)
    if missing_values_present:
        missing_indices = is_missing(values_to_add, mv)
        ups[nodes_to_update[missing_indices]] = mv
    return ups


@mask_and_unmask_data
def move_upstream(river_network, field, mv=np.nan, accept_missing=False):
    """
    Sets each node to be its downstream node value, or a missing value.

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
