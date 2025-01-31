from .core import flow
from .utils import mask_and_unmask_data
from ._catchment import _find_catchments_2D, _find_catchments_ND
import numpy as np

@mask_and_unmask_data
def find_catchments(river_network, field, mv=0, in_place=False):
    if not in_place:
        field = field.copy()

    if len(field.shape)==1:
        op = _find_catchments_2D
    else:
        op = _find_catchments_ND
    
    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, overwrite=True)

    flow(river_network, field, True, operation, mv)

@mask_and_unmask_data
def find_subcatchments(river_network, field, mv=0, in_place=False):
    if not in_place:
        field = field.copy()

    if len(field.shape)==1:
        op = _find_catchments_2D
    else:
        op = _find_catchments_ND
    
    def operation(river_network, field, grouping, mv):
        return op(river_network, field, grouping, mv, overwrite=False)

    flow(river_network, field, True, operation, mv)
