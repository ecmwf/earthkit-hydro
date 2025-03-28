import numpy as np

from .core import flow
from .utils import mask_and_unmask, missing_to_nan, nan_to_missing


@mask_and_unmask
def flow_downstream(
    river_network,
    field,
    mv=np.nan,
    in_place=False,
    ufunc=np.add,
    accept_missing=False,
    skip_missing_check=False,
    additive_weight=None,
    multiplicative_weight=None,
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
    skip_missing_check : bool, optional
        Whether to skip checking for missing values. Default is False.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """

    if not in_place:
        field = field.copy()

    field, field_dtype = missing_to_nan(field, mv, accept_missing, skip_missing_check)

    op = _ufunc_to_downstream

    def operation(
        river_network, field, grouping, mv, additive_weight, multiplicative_weight
    ):
        return op(
            river_network,
            field,
            grouping,
            mv,
            additive_weight,
            multiplicative_weight,
            ufunc=ufunc,
        )

    field = flow(
        river_network,
        field,
        False,
        operation,
        mv,
        additive_weight,
        multiplicative_weight,
    )

    return nan_to_missing(field, field_dtype, mv)


def _ufunc_to_downstream(
    river_network, field, grouping, mv, additive_weight, multiplicative_weight, ufunc
):
    """Updates field in-place by applying a ufunc at the downstream nodes of
    the grouping.

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
    if additive_weight is None:
        if multiplicative_weight is None:
            modifier_field = field[grouping]
        else:
            modifier_field = field[grouping] * multiplicative_weight[grouping]
    else:
        if multiplicative_weight is None:
            modifier_field = field[grouping] + additive_weight[grouping]
        else:
            modifier_field = (
                field[grouping] * multiplicative_weight[grouping]
                + additive_weight[grouping]
            )

    ufunc.at(
        field,
        (river_network.downstream_nodes[grouping], *[slice(None)] * (field.ndim - 1)),
        modifier_field,
    )


@mask_and_unmask
def flow_upstream(
    river_network,
    field,
    mv=np.nan,
    in_place=False,
    ufunc=np.add,
    accept_missing=False,
    skip_missing_check=False,
    additive_weight=None,
    multiplicative_weight=None,
):
    """Accumulates field values upstream.

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
    skip_missing_check : bool, optional
        Whether to skip checking for missing values. Default is False.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """

    if not in_place:
        field = field.copy()

    field, field_dtype = missing_to_nan(field, mv, accept_missing, skip_missing_check)

    op = _ufunc_to_upstream

    def operation(
        river_network, field, grouping, mv, additive_weight, multiplicative_weight
    ):
        return op(
            river_network,
            field,
            grouping,
            mv,
            additive_weight,
            multiplicative_weight,
            ufunc=ufunc,
        )

    field = flow(
        river_network,
        field,
        True,
        operation,
        mv,
        additive_weight,
        multiplicative_weight,
    )

    return nan_to_missing(field, field_dtype, mv)


def _ufunc_to_upstream(
    river_network, field, grouping, mv, additive_weight, multiplicative_weight, ufunc
):
    """Updates field in-place by applying a ufunc at the nodes of
    the grouping.

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
    down_group = river_network.downstream_nodes[grouping]
    if additive_weight is None:
        if multiplicative_weight is None:
            modifier_field = field[down_group]
        else:
            modifier_field = field[down_group] * multiplicative_weight[down_group]
    else:
        if multiplicative_weight is None:
            modifier_field = field[down_group] + additive_weight[down_group]
        else:
            modifier_field = (
                field[down_group] * multiplicative_weight[down_group]
                + additive_weight[down_group]
            )

    ufunc.at(
        field,
        (grouping, *[slice(None)] * (field.ndim - 1)),
        modifier_field,
    )
