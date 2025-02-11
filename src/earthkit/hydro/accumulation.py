import numpy as np
from .utils import mask_and_unmask_data, check_missing
from ._accumulation import (
    _ufunc_to_downstream,
    _ufunc_to_downstream_missing_values_2D,
    _ufunc_to_downstream_missing_values_ND,
)
from .core import flow


@mask_and_unmask_data
def flow_downstream(river_network, field, mv=np.nan, in_place=False, ufunc=np.add, accept_missing=False):
    """
    Accumulates field values downstream.

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
