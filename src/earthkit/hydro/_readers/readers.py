# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
    shape = x.shape
    x = x.flatten()
    missing_mask = x != -9999
    mask_upstream = ((x != -9) & (x != -9999)) & (x != -10)
    upstream_indices = np.arange(x.size)[mask_upstream]
    x = x[mask_upstream] - 1
    y = y.flatten()[mask_upstream] - 1
    downstream_indices = x + y * shape[1]
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


def from_d8(data, river_network_format="pcr_d8"):
    """
    Create a river network from PCRaster d8 data.

    Parameters
    ----------
    data : numpy.ndarray
        The PCRaster d8 drain direction data.

    Returns
    -------
    earthkit.hydro.network.RiverNetwork
        The created river network.
    """
    shape = data.shape
    data_flat = data.flatten()
    del data
    if river_network_format == "pcr_d8":
        missing_mask = np.isin(data_flat, range(1, 10))
        mask_upstream = data_flat != 5
    elif river_network_format == "esri_d8":
        missing_mask = np.isin(data_flat, np.append(0, 2 ** np.arange(8))) & (
            data_flat != 255
        )
        mask_upstream = (data_flat != 0) & (data_flat != -1)
    elif river_network_format == "merit_d8":
        missing_mask = np.isin(data_flat, np.append(0, 2 ** np.arange(8))) & (
            data_flat != 247
        )
        mask_upstream = (data_flat != 0) & (data_flat != 255)
    else:
        raise ValueError(f"Unsupported river network format: {river_network_format}.")
    mask_upstream = (mask_upstream) & (missing_mask)
    directions = data_flat[mask_upstream].astype("int")
    del data_flat
    if river_network_format == "pcr_d8":
        x_offsets = np.array([0, -1, 0, +1, -1, 0, +1, -1, 0, +1])[directions]
        y_offsets = -np.array([0, -1, -1, -1, 0, 0, 0, 1, 1, 1])[directions]
    elif river_network_format == "esri_d8" or river_network_format == "merit_d8":
        x_mapping = {32: -1, 64: 0, 128: +1, 16: -1, 1: +1, 8: -1, 4: 0, 2: +1}
        y_mapping = {32: 1, 64: 1, 128: 1, 16: 0, 1: 0, 8: -1, 4: -1, 2: -1}
        x_offsets = np.vectorize(x_mapping.get)(directions)
        y_offsets = -np.vectorize(y_mapping.get)(directions)
    del directions
    upstream_indices, downstream_indices = (
        find_upstream_downstream_indices_from_offsets(
            x_offsets, y_offsets, missing_mask, mask_upstream, shape
        )
    )
    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


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


def get_sources(n_nodes, down_ids):
    tmp_nodes = np.arange(n_nodes)
    tmp_nodes[down_ids] = n_nodes + 1
    inlets = tmp_nodes[tmp_nodes != n_nodes + 1]
    return inlets


def create_network(upstream_indices, downstream_indices, missing_mask, shape):
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

    edge_indices = np.arange(has_downstream.sum()).astype(np.uintp)

    up_ids = nodes[has_downstream].astype(np.uintp)
    down_ids = downstream[has_downstream].astype(np.uintp)

    mask = missing_mask.reshape(shape)
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
