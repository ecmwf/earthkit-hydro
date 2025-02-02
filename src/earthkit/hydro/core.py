def flow(river_network, field, invert_graph, operation, mv):
    if invert_graph:
        groupings = river_network.topological_groups[:-1][::-1]  # go from sinks to sources
    else:
        groupings = river_network.topological_groups[:-1]  # go from sources to sinks

    for grouping in groupings:
        # modify field in_place with desired operation
        # NB: this function needs to handle missing values
        # mv if they are allowed in input
        operation(river_network, field, grouping, mv)

    return field
