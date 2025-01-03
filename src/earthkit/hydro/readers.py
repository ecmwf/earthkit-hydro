import numpy as np
import xarray as xr
from .river_network import RiverNetwork
import joblib
from .caching import Cache
import tempfile


def load_river_network(
    domain,
    version,
    cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
    data_source="https://github.com/Oisin-M/river_network_store/raw/refs/heads/develop/{ekh_version}/{domain}/{version}/river_network.joblib",
    cache_fname="{ekh_version}_{domain}_{version}.joblib",
):
    """
    Loads a river network from a specified.
    A cache is used to store the river network file locally.

    Parameters
    ----------
    domain : str
        The domain identifier of the river network to load.
    version : str
        The version of the river network to load.
    cache_dir : str, optional
        Directory to store cached files (default is a temporary directory).
    data_source : str, optional
        URL template for downloading the river network file (default is a GitHub URL).
    cache_fname : str, optional
        Template for cache file name (default is "{ekh_version}_{domain}_{version}.joblib").

    Returns
    -------
    RiverNetwork
        The loaded river network object.
    """
    from ._version import __version__ as ekh_version

    cache = Cache(cache_dir, data_source, cache_fname)
    filepath = cache(ekh_version=ekh_version[0:3], domain=domain, version=version)
    network = joblib.load(filepath)

    return network


def from_netcdf_d8(filename):
    """
    Loads a river network from a NetCDF file using D8 direction encoding.

    Parameters
    ----------
    filename : str
        Path to the NetCDF file containing the D8 encoded data.

    Returns
    -------
    RiverNetwork
        The constructed river network object.
    """
    data = xr.open_dataset(filename, mask_and_scale=False)["Band1"].values
    return from_d8(data)


def from_d8(data):
    """
    Creates a river network from D8 direction data.

    Parameters
    ----------
    data : numpy.ndarray
        Array representing D8 directions.

    Returns
    -------
    RiverNetwork
        The constructed river network object.
    """
    shape = data.shape
    data_flat = data.flatten()
    del data

    mask_upstream = (data_flat != 255) & (data_flat != 5)
    missing_mask = data_flat != 255
    directions = data_flat[mask_upstream].astype("int")
    del data_flat

    x_offsets = np.array([0, -1, 0, +1, -1, 0, +1, -1, 0, +1])[directions]
    y_offsets = -np.array([0, -1, -1, -1, 0, 0, 0, 1, 1, 1])[directions]
    del directions

    upstream_indices, downstream_indices = find_upstream_downstream_indices_from_offsets(
        x_offsets, y_offsets, missing_mask, mask_upstream, shape
    )

    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def from_netcdf_cama(filename, type="nextxy"):
    """
    Loads a river network from a CaMa-Flood style NetCDF file.

    Parameters
    ----------
    filename : str
        Path to the NetCDF file.
    type : str, optional
        Type of CaMa-Flood encoding to use ("downxy" or "nextxy", default is "downxy").

    Returns
    -------
    RiverNetwork
        The constructed river network object.

    Raises
    ------
    Exception
        If an unknown type is specified.
    """
    data = xr.open_dataset(filename, mask_and_scale=False)
    if type == "downxy":
        dx, dy = data.downx.values, data.downy.values
        return from_cama_downxy(dx, dy)
    elif type == "nextxy":
        x, y = data.nextx.values, data.nexty.values
        return from_cama_nextxy(x, y)
    else:
        raise Exception("Unknown type")


def from_bin_cama(filename, type="downxy"):
    """
    Loads a river network from a CaMa-Flood style binary file.

    Parameters
    ----------
    filename : str
        Path to the binary file (without extension).
    type : str, optional
        Type of CaMa-Flood encoding to use ("downxy" or "nextxy", default is "downxy").

    Returns
    -------
    RiverNetwork
        The constructed river network object.

    Raises
    ------
    Exception
        If an unknown type is specified.
    """
    f = open(f"{filename}.ctl", "r")
    readfile = f.read()
    f.close()
    for line in readfile.splitlines():
        if "xdef" in line:
            split_line = line.split()
            assert split_line[0] == "xdef"
            nx = int(split_line[1])
        elif "ydef" in line:
            split_line = line.split()
            assert split_line[0] == "ydef"
            ny = int(split_line[1])
    data = np.fromfile(f"{filename}.bin", dtype=np.int32).reshape((nx, ny, 2), order="F")
    if type == "downxy":
        dx = data[:, :, 0].T
        dy = data[:, :, 1].T
        return from_cama_downxy(dx, dy)
    elif type == "nextxy":
        x = data[:, :, 0].T
        y = data[:, :, 1].T
        return from_cama_nextxy(x, y)
    else:
        raise Exception("Unknown type")


def from_cama_downxy(dx, dy):
    """
    Creates a river network from CaMa-Flood style downstream offsets (dx, dy).

    Parameters
    ----------
    dx : numpy.ndarray
        Array of x-direction offsets.
    dy : numpy.ndarray
        Array of y-direction offsets.

    Returns
    -------
    RiverNetwork
        The constructed river network object.
    """
    x_offsets = dx
    y_offsets = dy.flatten()
    shape = x_offsets.shape
    x_offsets = x_offsets.flatten()
    mask_upstream = ((x_offsets != -999) & (x_offsets != -9999)) & (x_offsets != -1000)
    missing_mask = x_offsets != -9999
    x_offsets = x_offsets[mask_upstream]
    y_offsets = y_offsets[mask_upstream]

    upstream_indices, downstream_indices = find_upstream_downstream_indices_from_offsets(
        x_offsets, y_offsets, missing_mask, mask_upstream, shape
    )

    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def from_cama_nextxy(x, y):
    """
    Creates a river network from CaMa-Flood style next x and y indices.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x indices for downstream nodes.
    y : numpy.ndarray
        Array of y indices for downstream nodes.

    Returns
    -------
    RiverNetwork
        The constructed river network object.
    """
    shape = x.shape
    x = x.flatten()
    missing_mask = x != -9999
    mask_upstream = ((x != -9) & (x != -9999)) & (x != -10)
    upstream_indices = np.arange(x.size)[mask_upstream]
    x = x[mask_upstream]
    y = y.flatten()[mask_upstream]

    x -= 1
    y -= 1

    downstream_indices = x + y * shape[1]

    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def find_upstream_downstream_indices_from_offsets(x_offsets, y_offsets, missing_mask, mask_upstream, shape):
    """
    Finds upstream and downstream indices from x and y offsets.

    Parameters
    ----------
    x_offsets : numpy.ndarray
        Array of x offsets.
    y_offsets : numpy.ndarray
        Array of y offsets.
    missing_mask : numpy.ndarray
        Mask indicating valid (non-missing) data points.
    mask_upstream : numpy.ndarray
        Mask indicating upstream data points.
    shape : tuple
        Shape of the data grid.

    Returns
    -------
    tuple of numpy.ndarray
        Arrays of upstream and downstream indices.
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


def create_network(upstream_indices, downstream_indices, missing_mask, shape):
    """
    Creates a river network from upstream and downstream indices.

    Parameters
    ----------
    upstream_indices : numpy.ndarray
        Array of upstream node indices.
    downstream_indices : numpy.ndarray
        Array of downstream node indices.
    missing_mask : numpy.ndarray
        Mask indicating valid (non-missing) data points.
    shape : tuple
        Shape of the data grid.

    Returns
    -------
    RiverNetwork
        The constructed river network object.
    """
    n_nodes = int(np.sum(missing_mask))
    nodes = np.arange(n_nodes, dtype=int)

    nodes_matrix = np.ones(missing_mask.size, dtype=int) * n_nodes
    nodes_matrix[missing_mask] = nodes

    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    del upstream_indices, downstream_indices, nodes_matrix

    downstream = np.ones(n_nodes, dtype=int) * n_nodes
    downstream[upstream_nodes] = downstream_nodes

    print("data loaded, initialising river network")

    return RiverNetwork(nodes, downstream, missing_mask.reshape(shape))
