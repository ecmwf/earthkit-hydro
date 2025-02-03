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
