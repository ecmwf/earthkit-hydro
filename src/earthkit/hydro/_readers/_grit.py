import numpy as np


def get_edge_indices(offsets, grouping):
    lengths = offsets[grouping + 1] - offsets[grouping]
    total_len = np.sum(lengths)
    result = np.empty(total_len, dtype=int)
    pos = 0

    for node, length in zip(grouping, lengths):
        start = offsets[node]
        for j in range(length):
            result[pos + j] = start + j
        pos += length
    return result


def compute_topological_labels_bifurcations(down_ids, offsets, sources, sinks):
    n_nodes = offsets.size - 1
    labels = np.zeros(n_nodes, dtype=int)
    inlets = sources

    for n in range(1, n_nodes + 1):
        inlets = np.unique(down_ids[get_edge_indices(offsets, inlets)])
        if inlets.size == 0:
            labels[sinks] = n - 1
            break
        labels[inlets] = n

    return labels
