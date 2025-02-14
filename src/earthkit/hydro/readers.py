import os
import tempfile
from hashlib import sha256
from io import BytesIO
from urllib.request import urlopen

import joblib
import numpy as np

from ._version import __version__ as ekh_version
from .river_network import RiverNetwork

# read in only up to second decimal point
# i.e. 0.1.dev90+gfdf4e33.d20250107 -> 0.1
ekh_version = ".".join(ekh_version.split(".")[:2])


def cache(func):
    """Decorator to allow automatic use of cache.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.

    """

    def wrapper(
        path,
        river_network_format,
        source="file",
        use_cache=True,
        cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
        cache_fname="{ekh_version}_{hash}.joblib",
        cache_compression=1,
    ):
        """Wrapper to load river network from cache if available, otherwise
        create and cache it.

        Parameters
        ----------
        path : str
            The path to the river network.
        river_network_format : str
            The format of the river network file.
            Supported formats are "precomputed", "cama", "pcr_d8", and "esri_d8".
        source : str, optional
            The source of the river network.
            For possible sources see:
            https://earthkit-data.readthedocs.io/en/latest/guide/sources.html
        use_cache : bool, optional
            Whether to use caching. Default is True.
        cache_dir : str, optional
            The directory to store the cache files. Default is a temporary directory.
        cache_fname : str, optional
            The filename template for the cache files.
            Default is "{ekh_version}_{hash}.joblib".
        cache_compression : int, optional
            The compression level for the cache files. Default is 1.

        Returns
        -------
        earthkit.hydro.RiverNetwork
            The loaded river network.

        """
        if use_cache:
            hashed_name = sha256(path.encode("utf-8")).hexdigest()
            cache_dir = cache_dir.format(ekh_version=ekh_version, hash=hashed_name)
            cache_fname = cache_fname.format(ekh_version=ekh_version, hash=hashed_name)
            cache_filepath = os.path.join(cache_dir, cache_fname)

            if os.path.isfile(cache_filepath):
                print(f"Loading river network from cache ({cache_filepath}).")
                return joblib.load(cache_filepath)
            else:
                print(f"River network not found in cache ({cache_filepath}).")
                os.makedirs(cache_dir, exist_ok=True)
        else:
            print("Cache disabled.")

        network = func(path, river_network_format, source)

        if use_cache:
            joblib.dump(network, cache_filepath, compress=cache_compression)
            print(f"River network loaded, saving to cache ({cache_filepath}).")

        return network

    return wrapper


def import_earthkit_or_prompt_install(river_network_format, source):
    """Ensure that the `earthkit.data` package is installed and import it. If
    the package is not installed, prompt the user to install it.

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
    """Find the main variable in the dataset.

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


@cache
def create_river_network(path, river_network_format, source):
    """Creates a river network from the given path, format, and source.

    Parameters
    ----------
    path : str
        The path to the river network data.
    river_network_format : str
        The format of the river network data.
        Supported formats are "precomputed", "cama", "pcr_d8", and "esri_d8".
    source : str
        The source of the river network data.
        For possible sources see:
        https://earthkit-data.readthedocs.io/en/latest/guide/sources.html.

    Returns
    -------
    object
        The river network object created from the given data.

    Raises
    ------
    ValueError
        If the river network format or source is unsupported.
    NotImplementedError
        If the river network format is "esri_d8".
    """
    if river_network_format == "precomputed":
        if source == "file":
            return joblib.load(path)
        elif source == "url":
            return joblib.load(BytesIO(urlopen(path).read()))
        else:
            raise ValueError(
                "Unsupported source for river network format"
                f"{river_network_format}: {source}."
            )
    elif river_network_format == "cama":
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        x, y = data.nextx.values, data.nexty.values
        return from_cama_nextxy(x, y)
    elif river_network_format == "pcr_d8":
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        var_name = find_main_var(data)
        return from_d8(data[var_name].values)
    elif river_network_format == "esri_d8":
        raise NotImplementedError(
            f"River network format {river_network_format} is not yet implemented."
        )
    else:
        raise ValueError(f"Unsupported river network format: {river_network_format}.")


def load_river_network(
    domain,
    river_network_version,
    data_source=(
        "https://github.com/Oisin-M/river_network_store/raw/refs/heads/develop/"
        "{ekh_version}/{domain}/{river_network_version}/river_network.joblib"
    ),
    *args,
    **kwargs,
):
    """Load a precomputed river network from a named domain and
    river_network_version.

    Parameters
    ----------
    domain : str
        The domain of the river network. Supported domains are "efas", "glofas", "cama".
    river_network_version : str
        The version of the river network on the specified domain.
    data_source : str, optional
        The data source URL template for the river network.
    *args : tuple
        Additional positional arguments to pass to `create_river_network`.
    **kwargs : dict
        Additional keyword arguments to pass to `create_river_network`.

    Returns
    -------
    earthkit.hydro.RiverNetwork
        The loaded river network.

    """
    uri = data_source.format(
        ekh_version=ekh_version,
        domain=domain,
        river_network_version=river_network_version,
    )
    return create_river_network(uri, "precomputed", "url", *args, **kwargs)


def from_cama_nextxy(x, y):
    """Create a river network from CaMa nextxy data.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates of the next downstream cell.
    y : numpy.ndarray
        The y-coordinates of the next downstream cell.

    Returns
    -------
    earthkit.hydro.RiverNetwork
        The created river network.

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


def from_cama_downxy(dx, dy):
    """Create a river network from CaMa downxy data.

    Parameters
    ----------
    x : numpy.ndarray
        The x-offsets of the next downstream cell.
    y : numpy.ndarray
        The y-offsets of the next downstream cell.

    Returns
    -------
    earthkit.hydro.RiverNetwork
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


def from_d8(data):
    """Create a river network from PCRaster d8 data.

    Parameters
    ----------
    data : numpy.ndarray
        The PCRaster d8 drain direction data.

    Returns
    -------
    earthkit.hydro.RiverNetwork
        The created river network.

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
    upstream_indices, downstream_indices = (
        find_upstream_downstream_indices_from_offsets(
            x_offsets, y_offsets, missing_mask, mask_upstream, shape
        )
    )
    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def find_upstream_downstream_indices_from_offsets(
    x_offsets, y_offsets, missing_mask, mask_upstream, shape
):
    """Function to convert from offsets to absolute indices.

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
    earthkit.hydro.RiverNetwork
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


def create_network(upstream_indices, downstream_indices, missing_mask, shape):
    """Creates a river network from upstream and downstream indices.

    Parameters
    ----------
    upstream_indices : numpy.ndarray
        Indices of upstream nodes.
    downstream_indices : numpy.ndarray
        Indices of downstream nodes.
    missing_mask : numpy.ndarray
        Boolean mask indicating the presence of nodes.
    shape : tuple
        Shape of the original domain.

    Returns
    -------
    earthkit.hydro.RiverNetwork
        The created river network.

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
    return RiverNetwork(nodes, downstream, missing_mask.reshape(shape))
