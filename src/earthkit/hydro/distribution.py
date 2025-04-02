import numpy as np

# @mask_and_unmask
# def get_upstream_field_vals(
#     river_network,
#     field,
#     metric,
#     stations_1d,
#     mv = np.nan,
#     accept_missing = False
# ):
#     field, field_dtype = missing_to_nan(field.copy(), mv, accept_missing)

#     weighted_field = field.copy()

#     ufunc = metrics_dict[metric].func

#     collect_upstream_points(river_network, weighted_field, ufunc, stations_1d)

#     return nan_to_missing(weighted_field, field_dtype, mv)


def collect_upstream_points(river_network):

    groupings = river_network.topological_groups[:-1]

    node_array = np.empty(river_network.n_nodes, dtype=object)
    node_array[river_network.nodes] = [[node] for node in river_network.nodes]

    for grouping in groupings:

        down_group = river_network.downstream_nodes[grouping]

        node_array[down_group] = (
            node_array[down_group] + node_array[grouping]
        )  # += doesn't work

    return node_array
