import numpy as np


def get_offsets(up_ids, n_nodes):
    counts = np.bincount(up_ids, minlength=n_nodes)
    offsets = np.zeros(n_nodes + 1, dtype=int)
    offsets[1:] = np.cumsum(counts)
    return offsets


def get_sinks_no_bifurcations(sinks, nodes, downstream_nodes, n_nodes):
    return sinks if sinks is not None else nodes[downstream_nodes == n_nodes]


def get_sinks_bifurcations(sinks, offsets):
    return np.where(offsets[1:] == offsets[:-1])[0] if sinks is None else sinks


def get_sources_no_bifurcations(sources, nodes, downstream_nodes, n_nodes):
    return (
        sources
        if sources is not None
        else _get_sources(nodes, downstream_nodes, n_nodes)
    )


def get_sources_bifurcations(sources, nodes, down_ids):
    return nodes[~np.isin(nodes, down_ids)] if sources is None else sources


def compute_topological_labels_bifurcations(down_ids, offsets, sources, sinks):
    from ._numba import get_edge_indices_numba

    n_nodes = offsets.size - 1
    labels = np.zeros(n_nodes, dtype=int)
    inlets = sources

    for n in range(1, n_nodes + 1):
        inlets = np.unique(down_ids[get_edge_indices_numba(offsets, inlets)])
        if inlets.size == 0:
            labels[sinks] = n - 1
            break
        labels[inlets] = n

    return labels


def _get_sources(nodes, downstream_nodes, n_nodes):
    """Identifies the source nodes in the river network (nodes with no
    upstream nodes).

    Returns
    -------
    numpy.ndarray
        Array of source nodes.

    """
    tmp_nodes = nodes.copy()
    downstream_no_sinks = downstream_nodes[
        downstream_nodes != n_nodes
    ]  # get all downstream nodes
    tmp_nodes[downstream_no_sinks] = (
        n_nodes + 1
    )  # downstream nodes that aren't sinks = -1
    inlets = tmp_nodes[
        tmp_nodes != n_nodes + 1
    ]  # sources are nodes that are not downstream nodes
    return inlets


def _find_new_masks(original_mask, mask):
    if mask.ndim == 1:
        river_network_mask = mask
        valid_indices = np.where(original_mask)
        new_valid_indices = (
            valid_indices[0][river_network_mask],
            valid_indices[1][river_network_mask],
        )
        domain_mask = np.full(original_mask.shape, False)
        domain_mask[new_valid_indices] = True
    else:
        domain_mask = mask & original_mask
        river_network_mask = domain_mask[original_mask]

    return domain_mask, river_network_mask


def _find_subnetwork_inputs(
    river_network_mask, original_downstream_nodes, original_n_nodes
):
    downstream_indices = original_downstream_nodes[river_network_mask]
    n_nodes = len(downstream_indices)  # number of nodes in the subnetwork
    # create new array of network nodes, setting all nodes not in mask to n_nodes
    subnetwork_nodes = np.full(original_n_nodes, n_nodes)
    subnetwork_nodes[river_network_mask] = np.arange(n_nodes)
    # get downstream nodes in the subnetwork
    non_sinks = downstream_indices != original_n_nodes
    downstream = np.full(n_nodes, n_nodes, dtype=np.uintp)
    downstream[non_sinks] = subnetwork_nodes[downstream_indices[non_sinks]]
    nodes = np.arange(n_nodes, dtype=np.uintp)

    sinks = nodes[downstream == n_nodes]
    sources = _get_sources(nodes, downstream, n_nodes)

    return nodes, downstream, n_nodes, sinks, sources
