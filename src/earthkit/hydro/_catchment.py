import numpy as np
from utils import is_missing

def _find_catchments_2D(river_network, field, grouping, mv, overwrite):
    valid_group = grouping[
        ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    ]  # only update nodes where the downstream belongs to a catchment
    if not overwrite:
        valid_group = valid_group[is_missing(field[valid_group], mv)]
    field[valid_group] = field[river_network.downstream_nodes[valid_group]]

def _find_catchments_ND(river_network, field, grouping, mv, overwrite):
    valid_mask = ~is_missing(field[river_network.downstream_nodes[grouping]], mv)
    valid_indices = np.array(np.where(valid_mask))
    valid_indices[0] = grouping[valid_indices[0]]
    if not overwrite:
        temp_valid_indices = valid_indices[0]
        valid_mask = is_missing(field[valid_indices], mv)
        valid_indices = np.array(np.where(valid_mask))
        valid_indices[0] = temp_valid_indices[valid_indices[0]]
    field[tuple(valid_indices)] = field[river_network.downstream_nodes[valid_indices]]
