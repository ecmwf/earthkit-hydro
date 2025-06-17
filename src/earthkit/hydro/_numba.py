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
