import numpy as np
from .utils import is_missing


def _ufunc_to_downstream(river_network, field, grouping, mv, ufunc):
    """
    Updates field in-place by applying a ufunc at the downstream nodes of the grouping, ignoring missing values.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        An array of indices.
    mv : scalar
        A missing value indicator (not used in the function but kept for consistency).
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html. Note: must allow two operands.

    Returns
    -------
    None
    """
    ufunc.at(field, river_network.downstream_nodes[grouping], field[grouping])


def _ufunc_to_downstream_missing_values_2D(river_network, field, grouping, mv, ufunc):
    """
    Applies a universal function (ufunc) to downstream nodes in a river network, dealing with missing values for 2D fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        An array of indices.
    mv : scalar
        A missing value indicator.
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html. Note: must allow two operands.

    Returns
    -------
    None
    """
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv))
    ufunc.at(field, nodes_to_update, values_to_add)
    field[nodes_to_update[missing_indices]] = mv


def _ufunc_to_downstream_missing_values_ND(river_network, field, grouping, mv, ufunc):
    """
    Applies a universal function (ufunc) to downstream nodes in a river network, dealing with missing values for ND fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        An array of indices.
    mv : scalar
        A missing value indicator.
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html. Note: must allow two operands.

    Returns
    -------
    None
    """
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv))
    ufunc.at(field, nodes_to_update, values_to_add)
    missing_indices = np.array(np.where(missing_indices))
    missing_indices[0] = nodes_to_update[missing_indices[0]]
    field[tuple(missing_indices)] = mv
