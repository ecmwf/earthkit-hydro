import numpy as np
from utils import is_missing

def _ufunc_to_downstream(river_network, field, grouping, mv, ufunc):
    ufunc.at(field, river_network.downstream_nodes[grouping], field[grouping])

def _ufunc_to_downstream_missing_values_2D(river_network, field, grouping, mv, ufunc):
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv))
    ufunc.at(field, nodes_to_update, values_to_add)
    field[nodes_to_update[missing_indices]] = mv

def _ufunc_to_downstream_missing_values_ND(river_network, field, grouping, mv, ufunc):
    nodes_to_update = river_network.downstream_nodes[grouping]
    values_to_add = field[grouping]
    missing_indices = np.logical_or(is_missing(values_to_add, mv), is_missing(field[nodes_to_update], mv))
    ufunc.at(field, nodes_to_update, values_to_add)
    missing_indices = np.array(np.where(missing_indices))
    missing_indices[0] = nodes_to_update[missing_indices[0]]
    field[tuple(missing_indices)] = mv