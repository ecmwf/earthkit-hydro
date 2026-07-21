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


def confluence_network():
    """A small branching network where sources ``{2, 3}`` merge at node 1 -> 0.

    Contributing (upstream) areas:  0->{0,1,2,3}, 1->{1,2,3}, 2->{2}, 3->{3}.
    Draining (downstream) areas:    0->{0}, 1->{0,1}, 2->{0,1,2}, 3->{0,1,3}.
    """
    return _network_from_nextxy([[-9, 1, 2, 2]], [[-9, 1, 1, 1]])


def gridded_network(flow_directions):
    """A network carrying grid coords, so the xarray wrapper can attach them."""
    river_network = _network_from_nextxy(*flow_directions)
    ny, nx = river_network.shape
    river_network.coords = {"y": np.arange(ny), "x": np.arange(nx)}
    return river_network


def to_dataarray(river_network, values):
    """Place a 1-D nodal array onto the network's 2-D grid as a DataArray."""
    import xarray as xr

    grid = np.zeros(river_network.shape, dtype=float)
    grid.flat[river_network.mask] = values
    return xr.DataArray(grid, dims=["y", "x"], coords=river_network.coords)


def make_field(river_network, seed=0):
    """A deterministic pseudo-random field defined over the network's nodes."""
    return np.random.default_rng(seed).standard_normal(river_network.n_nodes)
