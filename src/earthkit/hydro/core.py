import numpy as np

def flow(river_network, field, invert_graph, operation, mv):
    
    if invert_graph:
        groupings = river_network.topological_groups[:-1][::-1] # go from sinks to sources
    else:
        groupings = river_network.topological_groups[:-1] # go from sources to sinks

    for grouping in groupings:
        # modify field in_place with desired operation
        # NB: this function needs to handle missing values
        # mv if they are allowed in input
        operation(river_network, field, grouping, mv)
    
    return field

def _add(river_network, field, grouping):
    np.add.at(field, river_network.downstream_nodes[grouping], field[grouping])

def accuflux(river_network, field):
    flow(river_network, field, "down", _add)

def _set_value_if_unset(river_network, field, grouping):
    downstream_not_empty = field[river_network.downstream_nodes[grouping]]!=0
    valid_group = grouping[downstream_not_empty]
    field_to_update_empty = field[valid_group]==0
    valid_group = valid_group[field_to_update_empty]
    field[valid_group] = field[river_network.downstream_nodes[valid_group]]

def subcatchment(river_network, field):
    flow(river_network, field, "up", _set_value_if_unset)

def path(river_network, field):
    flow(river_network, field, "down", _set_value_if_unset)