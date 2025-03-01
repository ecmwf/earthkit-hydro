import numpy as np

from .core import flow
from .metrics import metrics_dict
from .utils import check_missing, is_missing, mask_and_unmask_data


@mask_and_unmask_data
def calculate_upstream_metric(
    river_network,
    field,
    metric,
    weights=None,
    mv=np.nan,
    in_place=False,
    accept_missing=False,
    missing_values_present_field=None,
    missing_values_present_weights=None,
):

    if weights is None:
        missing_values_present_weights = False
        weights = np.ones(river_network.n_nodes)
    else:
        field = (field.T * weights.T).T

    if missing_values_present_field is None:
        missing_values_present = check_missing(field, mv)
    else:
        missing_values_present = missing_values_present_field

    if missing_values_present_weights is None:
        missing_values_present_weights = check_missing(weights, mv)
        missing_values_present = (
            missing_values_present or missing_values_present_weights
        )
    else:
        missing_values_present = (
            missing_values_present or missing_values_present_weights
        )

    ufunc = metrics_dict[metric].func

    if missing_values_present and not np.isnan(mv):
        # TODO: handle missing values for mean
        raise NotImplementedError(
            "Support for generic missing values is not yet implemented."
        )

    field = flow_downstream(
        river_network,
        field,
        mv,
        in_place,
        ufunc,
        accept_missing,
        missing_values_present,
        skip=True,
    )

    if metric == "mean":
        counts = flow_downstream(
            river_network,
            weights,
            mv,
            in_place,
            ufunc,
            accept_missing,
            missing_values_present_weights,
            skip=True,
        )
        field_T = field.T
        field_T /= counts.T  # TODO: this does not handle arbitrary missing value logic
        return field_T.T
    else:
        return field


@mask_and_unmask_data
def flow_downstream(
    river_network,
    field,
    mv=np.nan,
    in_place=False,
    ufunc=np.add,
    accept_missing=False,
    missing_values_present=None,
):
    """Accumulates field values downstream.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    mv : scalar, optional
        The missing value indicator. Default is np.nan.
    in_place : bool, optional
        If True, modifies the input field in place. Default is False.
    ufunc : numpy.ufunc, optional
        The universal function (ufunc) to use for accumulation. Default is np.add.
    accept_missing : bool, optional
        If True, accepts missing values in the field. Default is False.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """
    if missing_values_present is None:
        missing_values_present = check_missing(field, mv, accept_missing)

    if not in_place:
        field = field.copy()

    if not missing_values_present or np.isnan(mv):
        op = _ufunc_to_downstream
    else:
        if len(field.shape) == 1:
            op = _ufunc_to_downstream_missing_values_2D
        else:
            op = _ufunc_to_downstream_missing_values_ND

    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, ufunc=ufunc)

    return flow(river_network, field, False, operation, mv)


def _ufunc_to_downstream(river_network, field, grouping, mv, ufunc):
    """Updates field in-place by applying a ufunc at the downstream nodes of
    the grouping, ignoring missing values.

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
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None

    """
    ufunc.at(
        field,
        (river_network.downstream_nodes[grouping], *[slice(None)] * (field.ndim - 1)),
        field[grouping],
    )


def _ufunc_to_downstream_missing_values_2D(river_network, field, grouping, mv, ufunc):
    """Applies a universal function (ufunc) to downstream nodes in a river
    network, dealing with missing values for 2D fields.

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
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None

    """
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(
        is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv)
    )
    ufunc.at(field, (nodes_to_update, *[slice(None)] * (field.ndim - 1)), values_to_add)
    field[nodes_to_update[missing_indices]] = mv


def _ufunc_to_downstream_missing_values_ND(river_network, field, grouping, mv, ufunc):
    """Applies a universal function (ufunc) to downstream nodes in a river
    network, dealing with missing values for ND fields.

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
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None

    """
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(
        is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv)
    )
    ufunc.at(field, (nodes_to_update, *[slice(None)] * (field.ndim - 1)), values_to_add)
    missing_indices = np.array(np.where(missing_indices))
    missing_indices[0] = nodes_to_update[missing_indices[0]]
    field[tuple(missing_indices)] = mv
