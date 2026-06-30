# ALGORITHM
# 1. Remove cycles - replace with missing.
# 2. Cells with a missing downstream are made into sinks.
# 3. NOT YET IMPLEMENTED. Cells pointing outside the domain are made into sinks
# 4. NOT YET IMPLEMENTED. Cells with invalid LDD values are made into sinks if a cell flows into it, otherwise set to missing.

import numpy as np

from earthkit.hydro._readers._cama import from_cama_nextxy_raw, load_cama_data
from earthkit.hydro._readers._d8 import from_d8_raw, load_d8_data
from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage


def set_sink_if_downstream_missing(up, down, mask, n_n, n_e, edge):
    invalid_nodes = down == n_n
    down = down[~invalid_nodes]
    up = up[~invalid_nodes]
    n_e = n_e - invalid_nodes.sum()
    edge = np.arange(n_e)
    return up, down, mask, n_n, n_e, edge


def set_missing_if_cycle(up, down, mask, n_n, n_e, edge):
    # DETECT CYCLES
    down_nodes = np.empty(n_n, dtype=int)
    down_nodes[up] = down

    current_nodes = up
    remaining_up_nodes = up
    nodes_in_cycles = np.array([], dtype=int)
    for _ in range(n_n):
        current_nodes = down_nodes[
            current_nodes
        ]  # node_ids downstream of current_nodes
        in_cycle = current_nodes == remaining_up_nodes
        nodes_in_cycles = np.append(nodes_in_cycles, current_nodes[in_cycle])
        remove_from_current = in_cycle | (current_nodes == n_n)
        current_nodes = current_nodes[~remove_from_current]
        remaining_up_nodes = remaining_up_nodes[~remove_from_current]
        if current_nodes.shape[0] == 0:
            break

    # REMOVE DETECTED CYCLES
    subset_of_mask = mask[mask]
    subset_of_mask[nodes_in_cycles] = False
    new_mask = mask.copy()
    new_mask[mask] = subset_of_mask
    lost_nodes = nodes_in_cycles.shape[0]
    n_n -= lost_nodes
    initial_cumsum = np.cumsum(mask) - 1
    new_cumsum = np.cumsum(new_mask) - 1
    new_cumsum = new_cumsum.reshape(new_mask.shape)
    new_cumsum[~new_mask] = n_n
    initial_cumsum = initial_cumsum.reshape(new_mask.shape)
    dictionary = dict(zip(initial_cumsum[mask].flatten(), new_cumsum[mask].flatten()))
    mapping = np.vectorize(dictionary.get)
    up = mapping(up)
    down = mapping(down)
    missing_nodes = up == n_n
    up = up[~missing_nodes]
    down = down[~missing_nodes]
    mask = new_mask
    n_e = up.shape[0]
    edge = np.arange(n_e)

    up, down, mask, n_n, n_e, edge = set_sink_if_downstream_missing(
        up, down, mask, n_n, n_e, edge
    )
    return up, down, mask, n_n, n_e, edge


def repair(path, river_network_format, source="file"):
    if river_network_format == "cama":
        data, coords = load_cama_data(path, river_network_format, source)
        up_ids, down_ids, edge_indices, mask, n_nodes, n_edges = from_cama_nextxy_raw(
            *data
        )
    elif (
        river_network_format == "pcr_d8"
        or river_network_format == "esri_d8"
        or river_network_format == "merit_d8"
    ):
        data, coords = load_d8_data(path, river_network_format, source)
        up_ids, down_ids, edge_indices, mask, n_nodes, n_edges = from_d8_raw(
            data, river_network_format=river_network_format
        )
    else:
        raise ValueError(
            f"Unsupported river network format for the repair method: {river_network_format}."
        )
    up, down, edge, mask, n_n, n_e = (
        up_ids,
        down_ids,
        edge_indices,
        mask,
        n_nodes,
        n_edges,
    )
    up, down, edge, mask, n_n, n_e = set_sink_if_downstream_missing(
        up, down, mask, n_n, n_e, edge
    )
    up, down, edge, mask, n_n, n_e = set_missing_if_cycle(
        up, down, edge, mask, n_n, n_e
    )

    store = RiverNetworkStorage(
        n_n,
        n_e,
        np.vstack([down, up, edge]).astype(np.int64),
        None,
        None,
        coords,
        None,
        None,
        np.where(mask.flatten())[0],
        mask.shape,
        False,
        None,
    )
    return store
