import numpy as np
from .utils import mask_2d
from .river_network import RiverNetwork


@mask_2d
def create_subnetwork(river_network, field, recompute=False):
    """
    Creates a subnetwork from the river network based on a mask.

    Parameters
    ----------
    field : numpy.ndarray
        A boolean mask to subset the river network.
    recompute : bool, optional
        If True, recomputes the topological labels for the subnetwork (default is False).

    Returns
    -------
    RiverNetwork
        A subnetwork of the river network.
    """
    river_network_mask = field
    valid_indices = np.where(river_network.mask)
    new_valid_indices = (valid_indices[0][river_network_mask], valid_indices[1][river_network_mask])
    domain_mask = np.full(river_network.mask.shape, False)
    domain_mask[new_valid_indices] = True

    downstream_indices = river_network.downstream_nodes[river_network_mask]
    n_nodes = len(downstream_indices)  # number of nodes in the subnetwork
    # create new array of network nodes, setting all nodes not in mask to n_nodes
    subnetwork_nodes = np.full(river_network.n_nodes, n_nodes)
    subnetwork_nodes[river_network_mask] = np.arange(n_nodes)
    # get downstream nodes in the subnetwork
    non_sinks = np.where(downstream_indices != river_network.n_nodes)
    downstream = np.full(n_nodes, n_nodes)
    downstream[non_sinks] = subnetwork_nodes[downstream_indices[non_sinks]]
    nodes = np.arange(n_nodes)

    if not recompute:
        sinks = nodes[downstream == n_nodes]
        topological_labels = river_network.topological_labels[river_network_mask]
        topological_labels[sinks] = river_network.n_nodes

        return RiverNetwork(nodes, downstream, domain_mask, sinks=sinks, topological_labels=topological_labels)
    else:
        return RiverNetwork(nodes, downstream, domain_mask)
