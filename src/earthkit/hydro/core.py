# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


def flow(river_network, field, invert_graph, operation, *args, **kwargs):
    """Apply an operation to a field along a river network.

    Parameters
    ----------
    river_network : earthkit.hydro.network_class.RiverNetwork
        An earthkit-hydro river network object.
    field : ndarray
        The field data to be modified in place.
    invert_graph : bool
        If True, process the river network from sinks to sources.
        If False, process from sources to sinks.
    operation : callable
        The operation to apply to the field. This function should
        take four arguments: river_network, field, grouping, and mv.
    *args, **kwargs : additional arguments
        Additional arguments to pass to the operation function.

    Returns
    -------
    ndarray
        The modified field after applying the operation along the river network.

    """
    # TODO: possibly switch logic between whether edge group or node group
    # dependening on if bifurcations or not
    if invert_graph:
        groupings = river_network.topological_groups_edges[
            ::-1
        ]  # go from sinks to sources
    else:
        groupings = river_network.topological_groups_edges

    for up_ids, down_ids in groupings:
        # modify field in_place with desired operation
        # NB: this function needs to handle missing values
        # mv if they are allowed in input
        operation(river_network, field, up_ids, down_ids, *args, **kwargs)

    return field
