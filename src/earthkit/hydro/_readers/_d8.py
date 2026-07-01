import numpy as np

from earthkit.hydro._utils.coords import get_core_grid_dims
from earthkit.hydro._utils.readers import from_file

from ._core import (
    create_initial_graph,
    create_network,
    find_main_var,
    find_upstream_downstream_indices_from_offsets,
    import_earthkit_or_prompt_install,
)


def load_d8_data(path, river_network_format, source="file"):
    if path.endswith(".map"):
        data = from_file(path, mask=False)
        coords = None
    else:
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        coord1, coord2 = get_core_grid_dims(data)
        coords = {
            coord1: data[coord1].values,
            coord2: data[coord2].values,
        }
        var_name = find_main_var(data)
        data = data[var_name].values
    return data, coords


def preprocess_d8_data(data, river_network_format="pcr_d8", truncate_domain=False):
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
            x_offsets, y_offsets, missing_mask, mask_upstream, shape, truncate_domain
        )
    )
    return upstream_indices, downstream_indices, missing_mask, shape


def from_d8_raw(data, river_network_format="pcr_d8"):
    upstream_indices, downstream_indices, missing_mask, shape = preprocess_d8_data(
        data, river_network_format, truncate_domain=True
    )
    up_ids, down_ids, edge_indices, mask, n_nodes, n_edges = create_initial_graph(
        upstream_indices, downstream_indices, missing_mask, shape
    )
    return up_ids, down_ids, edge_indices, mask, n_nodes, n_edges


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
    upstream_indices, downstream_indices, missing_mask, shape = preprocess_d8_data(
        data, river_network_format
    )
    return create_network(upstream_indices, downstream_indices, missing_mask, shape)
