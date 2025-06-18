# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from .utils import mask_and_unmask, missing_to_nan, nan_to_missing


@mask_and_unmask
def move_downstream(
    river_network,
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
    if not river_network.has_bifurcations:
        mask = (
            river_network.downstream_nodes != river_network.n_nodes
        )  # remove sinks since they have no downstream
        nodes_to_update = river_network.downstream_nodes[mask]
        values_to_add = (
            field[..., mask]
            if node_weights is None
            else field[..., mask] * node_weights[..., mask]
        )
    else:
        nodes_to_update = river_network.down_ids
        values_to_add = (
            field[..., river_network.up_ids]
            if node_weights is None
            else field[..., river_network.up_ids]
            * node_weights[..., river_network.up_ids]
        )

    if edge_weights is not None:
        values_to_add *= edge_weights

    ufunc.at(ups, (*[slice(None)] * (ups.ndim - 1), nodes_to_update), values_to_add)
    ups = nan_to_missing(ups, field_dtype, mv)
    return ups


@mask_and_unmask
def move_upstream(
    river_network,
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
    if not river_network.has_bifurcations:
        mask = river_network.downstream_nodes != river_network.n_nodes  # remove sinks
        update_vals = field[..., river_network.downstream_nodes[mask]]
        if node_weights is not None:
            update_vals *= node_weights[..., river_network.downstream_nodes[mask]]
        if edge_weights is not None:
            update_vals *= edge_weights
        down[..., mask] = update_vals
    else:
        nodes_to_update = river_network.up_ids
        update_vals = (
            field[..., river_network.down_ids]
            if node_weights is None
            else field[..., river_network.down_ids]
            * node_weights[..., river_network.down_ids]
        )
        if edge_weights is not None:
            update_vals *= edge_weights
        ufunc.at(down, (*[slice(None)] * (down.ndim - 1), nodes_to_update), update_vals)

    down = nan_to_missing(down, field_dtype, mv)

    return down
