import joblib
import tempfile
import os
import numpy as np
from urllib.request import urlopen
from hashlib import sha256
from io import BytesIO
from .river_network import RiverNetwork
from ._version import __version__ as ekh_version


# read in only up to second decimal point
# i.e. 0.1.dev90+gfdf4e33.d20250107 -> 0.1
ekh_version = ".".join(ekh_version.split(".")[:2])


def cache(func):
    def wrapper(
        path,
        river_format,
        source="file",
        use_cache=True,
        cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
        cache_fname="{ekh_version}_{hash}.joblib",
        cache_compression=1,
    ):
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

        network = func(path, river_format, source)

        if use_cache:
            joblib.dump(network, cache_filepath, compress=cache_compression)
            print(f"River network loaded, saving to cache ({cache_filepath}).")

        return network

    return wrapper


def import_earthkit_or_prompt_install(river_format, source):
    try:
        import earthkit.data as ekd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"earthkit-data is required for loading {river_format} from {source}.\nTo install it, run `pip install earthkit-data`"
        )
    return ekd


@cache
def create_river_network(path, river_format, source):
    if river_format == "precomputed":
        if source == "file":
            return joblib.load(path)
        elif source == "url":
            return joblib.load(BytesIO(urlopen(path).read()))
        else:
            raise ValueError(f"Unsupported source for format {river_format}: {source}.")
    elif river_format == "cama":
        ekd = import_earthkit_or_prompt_install(river_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        x, y = data.nextx.values, data.nexty.values
        return from_cama_nextxy(x, y)
    elif river_format == "pcr_d8":
        ekd = import_earthkit_or_prompt_install(river_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        return from_d8(data["Band1"].values)
    elif river_format == "esri_d8":
        raise NotImplementedError(f"River network format {river_format} is not yet implemented.")
    else:
        raise ValueError(f"Unsupported river network format: {river_format}.")


def load_river_network(
    domain,
    version,
    data_source="https://github.com/Oisin-M/river_network_store/raw/refs/heads/develop/{ekh_version}/{domain}/{version}/river_network.joblib",
    *args,
    **kwargs,
):
    uri = data_source.format(ekh_version=ekh_version[0:3], domain=domain, version=version)
    return create_river_network(uri, "precomputed", "url", *args, **kwargs)


def from_cama_nextxy(x, y):
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


def from_d8(data):
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


def find_upstream_downstream_indices_from_offsets(x_offsets, y_offsets, missing_mask, mask_upstream, shape):
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
