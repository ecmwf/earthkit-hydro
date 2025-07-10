import numpy as np

from earthkit.hydro.data_structures import RiverNetwork
from earthkit.hydro.utils.decorators import mask_and_unmask
from earthkit.hydro.utils.missing import missing_to_nan, nan_to_missing


@mask_and_unmask
def downstream(
    river_network: RiverNetwork,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    ufunc=np.add,
    accept_missing=False,
):
    """Sets each node to be the sum of its upstream nodes values, or a missing value.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
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
    if node_weights is not None:
        if field_dtype != node_weights.dtype:
            raise ValueError(
                f"""
                node_weights.dtype={node_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        node_weights, _ = missing_to_nan(node_weights.copy(), mv, accept_missing)
    if edge_weights is not None:
        if field_dtype != edge_weights.dtype:
            raise ValueError(
                f"""
                edge_weights.dtype={edge_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        edge_weights, _ = missing_to_nan(edge_weights.copy(), mv, accept_missing)

    ups = np.zeros(field.shape, dtype=field_dtype)

    did, uid, eid = river_network._storage.sorted_data
    values_to_add = (
        field[..., uid]
        if node_weights is None
        else field[..., uid] * node_weights[..., uid]
    )

    if edge_weights is not None:
        values_to_add *= edge_weights

    ufunc.at(ups, (*[slice(None)] * (ups.ndim - 1), did), values_to_add)
    ups = nan_to_missing(ups, field_dtype, mv)
    return ups


@mask_and_unmask
def upstream(
    river_network: RiverNetwork,
    field,
    node_weights=None,
    edge_weights=None,
    mv=np.nan,
    ufunc=np.add,
    accept_missing=False,
):
    """Sets each node to be its downstream node value, or a missing value.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
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

    if node_weights is not None:
        if field_dtype != node_weights.dtype:
            raise ValueError(
                f"""
                node_weights.dtype={node_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        node_weights, _ = missing_to_nan(node_weights.copy(), mv, accept_missing)
    if edge_weights is not None:
        if field_dtype != edge_weights.dtype:
            raise ValueError(
                f"""
                edge_weights.dtype={edge_weights.dtype} but field.dtype={field_dtype}.
                """
            )
        edge_weights, _ = missing_to_nan(edge_weights.copy(), mv, accept_missing)

    down = np.zeros(field.shape, dtype=field_dtype)

    did, uid, eid = river_network._storage.sorted_data
    update_vals = (
        field[..., did]
        if node_weights is None
        else field[..., did] * node_weights[..., did]
    )
    if edge_weights is not None:
        update_vals *= edge_weights
    ufunc.at(down, (*[slice(None)] * (down.ndim - 1), uid), update_vals)

    down = nan_to_missing(down, field_dtype, mv)

    return down
