def flow(river_network, field, invert_graph, operation, mv):
    """Apply an operation to a field along a river network.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : ndarray
        The field data to be modified in place.
    invert_graph : bool
        If True, process the river network from sinks to sources.
        If False, process from sources to sinks.
    operation : callable
        The operation to apply to the field. This function should
        take four arguments: river_network, field, grouping, and mv.
    mv : any
        The value representing missing data in the field.

    Returns
    -------
    ndarray
        The modified field after applying the operation along the river network.

    """
    if invert_graph:
        groupings = river_network.topological_groups[:-1][
            ::-1
        ]  # go from sinks to sources
    else:
        groupings = river_network.topological_groups[:-1]  # go from sources to sinks

    for grouping in groupings:
        # modify field in_place with desired operation
        # NB: this function needs to handle missing values
        # mv if they are allowed in input
        operation(river_network, field, grouping, mv)

    return field
