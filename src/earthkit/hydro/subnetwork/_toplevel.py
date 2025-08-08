import copy as cp

import numpy as np

from earthkit.hydro.data_structures import RiverNetwork


def create(river_network: RiverNetwork, node_mask=None, edge_mask=None, copy=True):
    """
    Create a subnetwork from a RiverNetwork.

    TODO: finish
    """
    if river_network.array_backend != "numpy" or copy is not True:
        raise NotImplementedError

    storage = cp.deepcopy(river_network._storage)
    valid_edges = edge_mask & (
        node_mask[storage.sorted_data[0]] & node_mask[storage.sorted_data[1]]
    )
    storage.sorted_data = storage.sorted_data[..., valid_edges]
    storage.splits = np.cumsum(valid_edges)[storage.splits - 1]
    storage.mask = storage.mask[node_mask]
    storage.n_nodes = storage.mask.shape[0]
    storage.n_edges = storage.sorted_data.shape[1]

    # TODO: add area and coords

    # TODO: decide if should be possible to shrink domain also

    return RiverNetwork(storage)
