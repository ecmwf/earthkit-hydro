import numba
import numpy as np


@numba.njit(numba.int64[:](numba.int64[:], numba.int64[:]))
def get_edge_indices_numba(offsets, grouping):
    lengths = offsets[grouping + 1] - offsets[grouping]
    total_len = np.sum(lengths)
    result = np.empty(total_len, dtype=np.int64)
    pos = 0

    for i in range(len(grouping)):
        node = grouping[i]
        length = lengths[i]
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
        inlets = np.unique(down_ids[get_edge_indices_numba(offsets, inlets)])
        if inlets.size == 0:
            labels[sinks] = n - 1
            break
        labels[inlets] = n

    return labels
