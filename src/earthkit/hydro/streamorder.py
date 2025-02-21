import numpy as np

from .core import flow
from .distance import compute_distance
from .river_network import RiverNetwork
from .utils import mask_and_unmask_data


@mask_and_unmask_data
def compute_streamorder(
    river_network,
    field,
    mv=-1,
    in_place=False,
):
    if not in_place:
        field = field.copy()
    field[river_network.sources] = 1

    distance_field = np.empty(river_network.n_nodes)
    distance_field.fill(-1)
    distance_field[river_network.sinks] = 0
    compute_distance(
        river_network, distance_field, in_place=True, allow_downstream=False
    )
    streamflow_river_network = RiverNetwork(
        river_network.nodes,
        river_network.downstream_nodes,
        river_network.mask,
        sinks=river_network.sinks,
        sources=river_network.sources,
        topological_labels=np.max(distance_field) - distance_field,
    )

    if len(field.shape) == 1:
        flow(streamflow_river_network, field, False, _compute_streamflow_2D, mv)
    # else:
    #     flow(river_network, field, True, _compute_streamflow_ND, mv)

    return field


def _compute_streamflow_2D(river_network, field, grouping, mv):
    indices = river_network.downstream_nodes[grouping]
    values = field[grouping]

    unique_indices, unique_index_positions = np.unique(indices, return_inverse=True)

    max_values_for_indices = np.zeros(len(unique_indices), dtype=np.float64)
    np.maximum.at(max_values_for_indices, unique_index_positions, values)

    mask = values == max_values_for_indices[unique_index_positions]

    count_max_value_for_indices = np.bincount(unique_index_positions, weights=mask)

    field[unique_indices] = max_values_for_indices + 1 * (
        count_max_value_for_indices > 1
    )
