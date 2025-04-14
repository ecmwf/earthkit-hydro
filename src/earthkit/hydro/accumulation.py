import numpy as np

from .core import flow
from .metrics import metrics_dict
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
    modifier_use_upstream=True,
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
    additive_weight : numpy.ndarray, optional
        A weight to be added to the field values. Default is None.
    multiplicative_weight : numpy.ndarray, optional
        A weight to be multiplied with the field values. Default is None.
    modifier_use_upstream : bool, optional
        If True, the modifiers are used on the upstream nodes instead of downstream.
        Default is True.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """

    if not in_place:
        field = field.copy()

    # TODO: decide if we should check all of the weights also
    field, field_dtype = missing_to_nan(field, mv, accept_missing, skip_missing_check)

    op = _ufunc_to_downstream

    def operation(
        river_network,
        field,
        grouping,
        mv,
        additive_weight,
        multiplicative_weight,
        modifier_use_upstream,
    ):
        return op(
            river_network,
            field,
            grouping,
            mv,
            additive_weight,
            multiplicative_weight,
            modifier_use_upstream,
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
        modifier_use_upstream,
    )

    return nan_to_missing(field, field_dtype, mv)


def _ufunc_to_downstream(
    river_network,
    field,
    grouping,
    mv,
    additive_weight,
    multiplicative_weight,
    modifier_use_upstream,
    ufunc,
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
    additive_weight : numpy.ndarray, optional
        A weight to be added to the field values. Default is None.
    multiplicative_weight : numpy.ndarray, optional
        A weight to be multiplied with the field values. Default is None.
    modifier_use_upstream : bool, optional
        If True, the modifiers are used on the upstream nodes instead of downstream.
        Default is True.
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None

    """
    modifier_group = (
        grouping if modifier_use_upstream else river_network.downstream_nodes[grouping]
    )

    if additive_weight is None:
        if multiplicative_weight is None:
            modifier_field = field[..., grouping]
        else:
            modifier_field = field[grouping] * multiplicative_weight[modifier_group]
    else:
        if multiplicative_weight is None:
            modifier_field = field[grouping] + additive_weight[modifier_group]
        else:
            modifier_field = (
                field[grouping] * multiplicative_weight[modifier_group]
                + additive_weight[modifier_group]
            )

    ufunc.at(
        field,
        (*[slice(None)] * (field.ndim - 1), river_network.downstream_nodes[grouping]),
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
    modifier_use_upstream=True,
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
    additive_weight : numpy.ndarray, optional
        A weight to be added to the field values. Default is None.
    multiplicative_weight : numpy.ndarray, optional
        A weight to be multiplied with the field values. Default is None.
    modifier_use_upstream : bool, optional
        If True, the modifiers are used on the upstream nodes instead of downstream.
        Default is True.

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
        river_network,
        field,
        grouping,
        mv,
        additive_weight,
        multiplicative_weight,
        modifier_use_upstream,
    ):
        return op(
            river_network,
            field,
            grouping,
            mv,
            additive_weight,
            multiplicative_weight,
            modifier_use_upstream,
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
        modifier_use_upstream,
    )

    return nan_to_missing(field, field_dtype, mv)


def _ufunc_to_upstream(
    river_network,
    field,
    grouping,
    mv,
    additive_weight,
    multiplicative_weight,
    modifier_use_upstream,
    ufunc,
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
    additive_weight : numpy.ndarray, optional
        A weight to be added to the field values. Default is None.
    multiplicative_weight : numpy.ndarray, optional
        A weight to be multiplied with the field values. Default is None.
    modifier_use_upstream : bool, optional
        If True, the modifiers are used on the upstream nodes instead of downstream.
        Default is True.
    ufunc : numpy.ufunc
        A universal function from the numpy library to be applied to the field data.
        Available ufuncs: https://numpy.org/doc/2.2/reference/ufuncs.html.
        Note: must allow two operands.

    Returns
    -------
    None

    """
    down_group = river_network.downstream_nodes[grouping]
    modifier_group = (
        grouping if modifier_use_upstream else river_network.downstream_nodes[grouping]
    )
    if additive_weight is None:
        if multiplicative_weight is None:
            modifier_field = field[..., down_group]
        else:
            modifier_field = (
                field[..., down_group] * multiplicative_weight[..., modifier_group]
            )
    else:
        if multiplicative_weight is None:
            modifier_field = (
                field[..., down_group] + additive_weight[..., modifier_group]
            )
        else:
            modifier_field = (
                field[..., down_group] * multiplicative_weight[..., modifier_group]
                + additive_weight[..., modifier_group]
            )

    ufunc.at(
        field,
        (*[slice(None)] * (field.ndim - 1), grouping),
        modifier_field,
    )


def calculate_online_metric(
    river_network, field, metric, weights, mv, accept_missing, flow_direction
):

    if flow_direction == "up":
        flow_func = flow_upstream
    elif flow_direction == "down":
        flow_func = flow_downstream

    """
    Calculates a metric for the field over all upstream or downstream values.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum", "product"
    weights : ndarray
        Used to weight the field when computing the metric.
    mv : scalar
        Missing value for the input field.
    accept_missing : bool
        Whether or not to accept missing values in the input field.
    flow_direction : str
        The direction of flow for the metric calculation. Options are "up" or "down".

    Returns
    -------
    numpy.ndarray
        Output field.

    """

    field, field_dtype = missing_to_nan(field.copy(), mv, accept_missing)

    if weights is None:
        if metric == "mean" or metric == "std" or metric == "var":
            weightings = np.ones(river_network.n_nodes, dtype=np.float64)
        weighted_field = field.copy()
    else:
        assert field_dtype == weights.dtype
        weightings, _ = missing_to_nan(weights.copy(), mv, accept_missing)
        weighted_field = field * weightings  # this isn't in_place !

    ufunc = metrics_dict[metric].func

    weighted_field = flow_func(
        river_network,
        weighted_field,
        np.nan,  # mv replaced by nan
        True,  # do in-place on field copy
        ufunc,
        accept_missing,
        skip_missing_check=True,
        skip=True,
    )

    if metric == "mean" or metric == "std" or metric == "var":
        counts = flow_func(
            river_network,
            weightings,
            np.nan,  # mv replaced by nan
            False,
            ufunc,
            accept_missing,
            skip_missing_check=True,
            skip=True,
        )

        if metric == "mean":
            weighted_field /= counts  # weighted mean
            return nan_to_missing(
                weighted_field, np.float64, mv
            )  # if we compute means, we change dtype for int fields etc.
        elif metric == "var" or metric == "std":
            weighted_sum_of_squares = flow_func(
                river_network,
                field**2 * weightings if weights is not None else field**2,
                np.nan,  # mv replaced by nan
                True,  # do in-place on field copy
                ufunc,
                accept_missing,
                skip_missing_check=True,
                skip=True,
            )
            mean = weighted_field / counts
            weighted_sum_of_squares = weighted_sum_of_squares / counts - mean**2
            weighted_sum_of_squares[weighted_sum_of_squares < 0] = (
                0  # can occur for numerical issues
            )
            if metric == "var":
                return nan_to_missing(weighted_sum_of_squares, np.float64, mv)
            elif metric == "std":
                return nan_to_missing(np.sqrt(weighted_sum_of_squares), np.float64, mv)

    else:
        return nan_to_missing(weighted_field, field_dtype, mv)
