import numpy as np

from earthkit.hydro._utils.coords import get_core_grid_dims

from ._core import (
    create_initial_graph,
    create_network,
    find_upstream_downstream_indices_from_offsets,
    import_earthkit_or_prompt_install,
)


def load_cama_data(path, river_network_format, source="file"):
    ekd = import_earthkit_or_prompt_install(river_network_format, source)
    data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
    x, y = data.nextx.values, data.nexty.values
    coord1, coord2 = get_core_grid_dims(data)
    coords = {
        coord1: data[coord1].values,
        coord2: data[coord2].values,
    }
    return (x, y), coords


def from_cama_nextxy(x, y):
    """
    Create a river network from CaMa nextxy data.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the next downstream cell.
    y : numpy.ndarray
        The y-coordinates of the next downstream cell.

    Returns
    -------
    earthkit.hydro.network.RiverNetwork
        The created river network.
    """
    upstream_indices, downstream_indices, missing_mask, shape = (
        preprocess_cama_nextxy_data(x, y)
    )
    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def from_cama_downxy(dx, dy):
    """
    Create a river network from CaMa downxy data.

    Parameters
    ----------
    dx : numpy.ndarray
        The x-offsets of the next downstream cell.
    dy : numpy.ndarray
        The y-offsets of the next downstream cell.

    Returns
    -------
    earthkit.hydro.network.RiverNetwork
        The created river network.
    """
    x_offsets = dx
    y_offsets = dy.flatten()
    shape = x_offsets.shape
    x_offsets = x_offsets.flatten()
    mask_upstream = ((x_offsets != -999) & (x_offsets != -9999)) & (x_offsets != -1000)
    missing_mask = x_offsets != -9999
    x_offsets = x_offsets[mask_upstream]
    y_offsets = y_offsets[mask_upstream]
    upstream_indices, downstream_indices = (
        find_upstream_downstream_indices_from_offsets(
            x_offsets, y_offsets, missing_mask, mask_upstream, shape
        )
    )
    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def preprocess_cama_nextxy_data(x, y):
    shape = x.shape
    x = x.flatten()
    missing_mask = x != -9999
    mask_upstream = ((x != -9) & (x != -9999)) & (x != -10)
    upstream_indices = np.arange(x.size)[mask_upstream]
    x = x[mask_upstream] - 1
    y = y.flatten()[mask_upstream] - 1
    downstream_indices = x + y * shape[1]
    return upstream_indices, downstream_indices, missing_mask, shape


def from_cama_nextxy_raw(x, y):
    upstream_indices, downstream_indices, missing_mask, shape = (
        preprocess_cama_nextxy_data(x, y)
    )
    up_ids, down_ids, edge_indices, mask, n_nodes, n_edges = create_initial_graph(
        upstream_indices, downstream_indices, missing_mask, shape
    )
    return up_ids, down_ids, edge_indices, mask, n_nodes, n_edges
