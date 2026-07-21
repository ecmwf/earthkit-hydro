import numpy as np


def extend_array(arr, extra_shape):
    current_shape = arr.shape
    new_shape = (*extra_shape, *current_shape)
    return np.broadcast_to(arr, new_shape)


def convert_to_2d(river_network, array, fill_value):
    field = np.full(river_network.mask.shape, fill_value=fill_value, dtype=array.dtype)
    field[river_network.mask] = array
    return field


def _network_from_nextxy(x, y):
    from earthkit.hydro._readers import from_cama_nextxy
    from earthkit.hydro.data_structures import RiverNetwork

    return RiverNetwork(from_cama_nextxy(np.asarray(x), np.asarray(y)))


def star_network(n):
    """Sink node 0 with nodes ``1..n-1`` draining directly into it.

    Node 0's contributing area is the whole network; every other node is a leaf.
    """
    x = [[-9] + [1] * (n - 1)]  # every non-sink cell drains to column 1 (node 0)
    y = [[-9] + [1] * (n - 1)]
    return _network_from_nextxy(x, y)


def chain_network(n):
    """A single chain ``0 <- 1 <- ... <- n-1`` (node ``i`` drains to node ``i-1``).

    The contributing area of node ``k`` is exactly ``{k, ..., n-1}`` and its
    draining area is ``{0, ..., k}``, so every node's aggregation set is a known
    contiguous slice of the field.
    """
    x = [[-9] + list(range(1, n))]  # cell i (i>=1) drains to column i (node i-1)
    y = [[-9] + [1] * (n - 1)]
    return _network_from_nextxy(x, y)
