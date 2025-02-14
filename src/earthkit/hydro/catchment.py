import numpy as np

from .core import flow
from .utils import is_missing, mask_and_unmask_data


@mask_and_unmask_data
def find_catchments(river_network, field, mv=0, in_place=False):
    """
    Labels the catchments given a field of labelled sinks.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    mv : scalar, optional
        The missing value indicator. Default is 0.
    in_place : bool, optional
        If True, modifies the input field in place. Default is False.
    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.
    """
    if not in_place:
        field = field.copy()

    if len(field.shape) == 1:
        op = _find_catchments_2D
    else:
        op = _find_catchments_ND

    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, overwrite=True)

    return flow(river_network, field, True, operation, mv)


@mask_and_unmask_data
def find_subcatchments(river_network, field, mv=0, in_place=False):
    """
    Labels the subcatchments given a field of labelled sinks.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    mv : scalar, optional
        The missing value indicator. Default is 0.
    in_place : bool, optional
        If True, modifies the input field in place. Default is False.
    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.
    """
    if not in_place:
        field = field.copy()

    if len(field.shape) == 1:
        op = _find_catchments_2D
    else:
        op = _find_catchments_ND

    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, overwrite=False)

    return flow(river_network, field, True, operation, mv)


def _find_catchments_2D(river_network, field, grouping, mv, overwrite):
    """
    Updates field in-place with the value of its downstream nodes, dealing with missing values for 2D fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    mv : scalar
        The missing value indicator.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None
    """
    valid_group = grouping[
        ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    ]  # only update nodes where the downstream belongs to a catchment
    if not overwrite:
        valid_group = valid_group[is_missing(field[valid_group], mv)]
    field[valid_group] = field[river_network.downstream_nodes[valid_group]]


def _find_catchments_ND(river_network, field, grouping, mv, overwrite):
    """
    Updates field in-place with the value of its downstream nodes, dealing with missing values for ND fields.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    grouping : numpy.ndarray
        The array of node indices.
    mv : scalar
        The missing value indicator.
    overwrite : bool
        If True, overwrite existing non-missing values in the field array.

    Returns
    -------
    None
    """
    valid_mask = ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    valid_indices = np.array(np.where(valid_mask))
    valid_indices[0] = grouping[valid_indices[0]]
    if not overwrite:
        temp_valid_indices = valid_indices[0]
        valid_mask = is_missing(field[valid_indices], mv)
        valid_indices = np.array(np.where(valid_mask))
        valid_indices[0] = temp_valid_indices[valid_indices[0]]
    field[tuple(valid_indices)] = field[river_network.downstream_nodes[valid_indices]]
