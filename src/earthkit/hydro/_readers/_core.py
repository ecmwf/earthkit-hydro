import numpy as np

from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage

from .group_labels import compute_topological_labels


def import_earthkit_or_prompt_install(river_network_format, source):
    """
    Ensure that the `earthkit.data` package is installed and import it.
    If the package is not installed, prompt the user to install it.

    Parameters
    ----------
    river_network_format : str
        The format of the river network file.
    source : str
        The source of the river network.

    Returns
    -------
    module
        The imported `earthkit.data` module.

    Raises
    ------
    ModuleNotFoundError
        If the `earthkit.data` package is not installed.
    """
    try:
        import earthkit.data as ekd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "earthkit-data is required for loading river network format"
            f"{river_network_format} from source {source}."
            "\nTo install it, run `pip install earthkit-data`"
        )
    return ekd


def find_main_var(ds, min_dim=2):
    """
    Find the main variable in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to search for the main variable.
    min_dim : int, optional
        The minimum number of dimensions the variable must have. Default is 2.

    Returns
    -------
    str
        The name of the main variable.

    Raises
    ------
    ValueError
        If no variable or more than one variable with the required dimensions is found.
    """
    variable_names = [k for k in ds.variables if len(ds.variables[k].dims) >= min_dim]
    if len(variable_names) > 1:
        raise ValueError("More than one variable of dimension >= {min_dim} in dataset.")
    elif len(variable_names) == 0:
        raise ValueError("No variable of dimension >= {min_dim} in dataset.")
    else:
        return variable_names[0]


def find_upstream_downstream_indices_from_offsets(
    x_offsets, y_offsets, missing_mask, mask_upstream, shape
):
    """
    Function to convert from offsets to absolute indices.

    Parameters
    ----------
    x_offsets : numpy.ndarray
        The x-offsets of the next downstream cell.
    y_offsets : numpy.ndarray
        The y-offsets of the next downstream cell.
    missing_mask : numpy.ndarray
        A boolean mask indicating missing values in the data.
    mask_upstream : numpy.ndarray
        A boolean mask indicating upstream cells.
    shape : tuple
        The shape of the original data array.

    Returns
    -------
    earthkit.hydro.network.RiverNetwork
        The created river network.
    """
    ny, nx = shape
    upstream_indices = np.arange(missing_mask.size)[mask_upstream]
    del mask_upstream
    x_coords = upstream_indices % nx
    x_coords = (x_coords + x_offsets) % nx
    downstream_indices = x_coords
    del x_coords
    y_coords = np.floor_divide(upstream_indices, nx)
    y_coords = (y_coords + y_offsets) % ny
    downstream_indices += y_coords * nx
    del y_coords
    return upstream_indices, downstream_indices


def create_graph_nodes_edges(upstream_indices, downstream_indices, missing_mask):

    n_nodes = int(np.sum(missing_mask))
    nodes = np.arange(n_nodes, dtype=np.uintp)
    nodes_matrix = np.full(missing_mask.size, n_nodes, dtype=np.uintp)
    nodes_matrix[missing_mask] = nodes
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    del upstream_indices, downstream_indices, nodes_matrix
    downstream = np.full(n_nodes, n_nodes, dtype=np.uintp)
    downstream[upstream_nodes] = downstream_nodes
    del downstream_nodes, upstream_nodes

    has_downstream = downstream != n_nodes

    n_edges = int(has_downstream.sum())

    return nodes, downstream, has_downstream, n_nodes, n_edges


def get_up_down_edge_mask(has_downstream, nodes, downstream, missing_mask, shape):

    edge_indices = np.arange(has_downstream.sum()).astype(np.uintp)
    up_ids = nodes[has_downstream].astype(np.uintp)
    down_ids = downstream[has_downstream].astype(np.uintp)

    mask = missing_mask.reshape(shape)
    return up_ids, down_ids, edge_indices, mask


def get_sources(n_nodes, down_ids):
    tmp_nodes = np.arange(n_nodes)
    tmp_nodes[down_ids] = n_nodes + 1
    inlets = tmp_nodes[tmp_nodes != n_nodes + 1]
    return inlets


def create_initial_graph(upstream_indices, downstream_indices, missing_mask, shape):
    nodes, downstream, has_downstream, n_nodes, n_edges = create_graph_nodes_edges(
        upstream_indices, downstream_indices, missing_mask
    )
    up_ids, down_ids, edge_indices, mask = get_up_down_edge_mask(
        has_downstream, nodes, downstream, missing_mask, shape
    )
    return up_ids, down_ids, edge_indices, mask, n_nodes, n_edges


def create_network(upstream_indices, downstream_indices, missing_mask, shape):

    nodes, downstream, has_downstream, n_nodes, n_edges = create_graph_nodes_edges(
        upstream_indices, downstream_indices, missing_mask
    )
    up_ids, down_ids, edge_indices, mask = get_up_down_edge_mask(
        has_downstream, nodes, downstream, missing_mask, shape
    )

    bifurcates = False
    sources = get_sources(n_nodes, down_ids)
    sinks = nodes[downstream == n_nodes]

    coords = None

    assert np.all(np.isin(np.setdiff1d(sinks, sources), down_ids))

    distances = compute_topological_labels(
        sources.astype(np.uintp),
        sinks.astype(np.uintp),
        downstream.astype(np.uintp),
        n_nodes,
    )[has_downstream]

    sort_indices = np.lexsort(
        (nodes[has_downstream], distances)
    )  # np.argsort(distances)
    sorted_distances = distances[sort_indices]  # from source to sink

    up_ids_sort = up_ids[sort_indices]
    down_ids_sort = down_ids[sort_indices]
    edge_ids_sort = edge_indices[sort_indices]

    _, splits = np.unique(sorted_distances, return_index=True)
    splits = splits[1:]

    pixarea = None
    edge_weights = None

    store = RiverNetworkStorage(
        n_nodes,
        n_edges,
        np.vstack([down_ids_sort, up_ids_sort, edge_ids_sort]).astype(np.int64),
        sources,
        sinks,
        coords,
        splits,
        pixarea,
        np.where(mask.flatten())[0],
        mask.shape,
        bifurcates,
        edge_weights,
    )

    return store


def assign_coords(river_network_storage, data, coords):
    if coords is not None:
        river_network_storage.coords = coords
    # else warn
    return river_network_storage
