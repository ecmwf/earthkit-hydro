import copy as cp

from earthkit.hydro._backends.numpy_backend import NumPyBackend
from earthkit.hydro._utils.decorators.masking import mask_last2_dims
from earthkit.hydro.data_structures import RiverNetwork

np = NumPyBackend()


def from_mask(river_network: RiverNetwork, node_mask=None, edge_mask=None, copy=True):
    """
    Create a subnetwork from a river network.

    Parameters
    ----------
    river_network : RiverNetwork
        Original river network from which to create a subnetwork.
    node_mask : array, optional
        A mask of the network nodes or gridcells. Default is None (all True).
    edge_mask : array, optional
        A mask of the network edges. Default is None (all True).
    copy : bool, optional
        Whether or not to modify the original river network or return a copy. Default is True.

    Returns
    -------
    RiverNetwork
        The river network object created from the given data.
    """
    if river_network.array_backend != "numpy" or copy is not True:
        raise NotImplementedError

    if node_mask is None and edge_mask is None:
        return cp.deepcopy(river_network)

    if node_mask is not None:
        if node_mask.shape[-2:] == river_network.shape:
            node_mask = mask_last2_dims(
                np, node_mask, river_network.mask, node_mask.shape
            )

    node_relabel = np.empty(river_network.n_nodes, dtype=int)
    node_relabel[node_mask] = np.arange(node_mask.sum())

    storage = cp.deepcopy(river_network._storage)
    if edge_mask is not None and node_mask is not None:
        valid_edges = edge_mask[storage.sorted_data[2]] & (
            node_mask[storage.sorted_data[0]] & node_mask[storage.sorted_data[1]]
        )
    elif edge_mask is None:
        valid_edges = (
            node_mask[storage.sorted_data[0]] & node_mask[storage.sorted_data[1]]
        )
    else:
        valid_edges = edge_mask[storage.sorted_data[2]]

    original_order_edge_mask = np.empty(river_network.n_edges, dtype=bool)
    original_order_edge_mask[storage.sorted_data[2]] = valid_edges
    edge_relabel = np.empty(river_network.n_edges, dtype=int)
    edge_relabel[original_order_edge_mask] = np.arange(original_order_edge_mask.sum())

    storage.sorted_data = storage.sorted_data[..., valid_edges]
    storage.sorted_data[0] = node_relabel[storage.sorted_data[0]]
    storage.sorted_data[1] = node_relabel[storage.sorted_data[1]]
    storage.sorted_data[2] = edge_relabel[storage.sorted_data[2]]

    storage.splits = np.cumsum(valid_edges)[storage.splits - 1]
    storage.mask = storage.mask[node_mask]
    storage.n_nodes = storage.mask.shape[0]
    storage.n_edges = storage.sorted_data.shape[1]

    return RiverNetwork(storage)


def crop(river_network, copy=True):
    """
    Crop a gridded network to the minimum bounding grid.

    Parameters
    ----------
    river_network : RiverNetwork
        Original river network from which to create a cropped network.
    copy : bool, optional
        Whether or not to modify the original river network or return a copy. Default is True.

    Returns
    -------
    RiverNetwork
        The river network object created from the given data.
    """

    if river_network.array_backend != "numpy" or copy is not True:
        raise NotImplementedError

    storage = cp.deepcopy(river_network._storage)

    rows, cols = np.unravel_index(storage.mask, shape=(storage.shape))

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    storage.shape = (int(row_max - row_min + 1), int(col_max - col_min + 1))

    storage.mask = np.ravel_multi_index(
        (rows - row_min, cols - col_min), dims=storage.shape
    )

    for i, key in enumerate(storage.coords.keys()):
        if i == 0:
            storage.coords[key] = storage.coords[key][row_min : row_max + 1]
        elif i == 1:
            storage.coords[key] = storage.coords[key][col_min : col_max + 1]
        else:
            raise ValueError("coords must not have more than 2 keys.")

    return RiverNetwork(storage)
