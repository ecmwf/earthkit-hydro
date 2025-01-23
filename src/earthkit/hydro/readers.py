import numpy as np
from .river_network import RiverNetwork
import joblib
import tempfile
import os
from urllib.request import urlretrieve, urlopen
from io import BytesIO


def load_river_network(
    name,
    version=None,
    use_cache=True,
    cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
    data_source="https://github.com/Oisin-M/river_network_store/raw/refs/heads/develop/{ekh_version}/{name}/{version}/river_network.joblib",
    cache_fname="{ekh_version}_{name}_{version}.joblib",
    cache_compression=1,
):
    from ._version import __version__ as ekh_version

    if version is None:
        sanitised_name = name.replace("/", "").replace(".", "")
    else:
        sanitised_name = name

    if use_cache:
        cache_dir = cache_dir.format(ekh_version=ekh_version[0:3], name=sanitised_name, version=version)
        cache_fname = cache_fname.format(ekh_version=ekh_version[0:3], name=sanitised_name, version=version)
        cache_filepath = os.path.join(cache_dir, cache_fname)

        if os.path.isfile(cache_filepath):
            print(f"Loading river network from cache ({cache_filepath}).")
            return joblib.load(cache_filepath)
        else:
            print(f"River network not found in cache ({cache_filepath}).")
            os.makedirs(cache_dir, exist_ok=True)
            if version is None:
                network = load_from_file(name)
                joblib.dump(network, cache_filepath, compress=cache_compression)
                print(f"River network loaded, saving to cache ({cache_filepath}).")
                return network
            else:
                url = data_source.format(ekh_version=ekh_version[0:3], name=name, version=version)
                urlretrieve(url, cache_filepath)
                print(f"River network loaded, saving to cache ({cache_filepath}).")
                return joblib.load(cache_filepath)
    else:
        print("Warning: cache disabled. Loading a river network may be slow.")
        if version is None:
            network = load_from_file(name)
            print("River network loaded.")
            return network
        else:
            url = data_source.format(ekh_version=ekh_version[0:3], name=name, version=version)
            network = joblib.load(BytesIO(urlopen(url).read()))
            print("River network loaded.")
            return network


def load_from_file(filename):
    if filename.endswith(".joblib") or filename.endswith(".pkl") or filename.endswith(".pickle"):
        return joblib.load(filename)
    elif filename.endswith(".bin"):
        data = load_cama_binfile(filename)
        if np.any((data < 0) & ((data != -9) & (data != -9999)) & (data != -10)):
            x = data[:, :, 0].T
            y = data[:, :, 1].T
            return from_cama_nextxy(x, y)
        else:
            dx = data[:, :, 0].T
            dy = data[:, :, 1].T
            return from_cama_downxy(dx, dy)
    elif filename.endswith(".nc") or filename.endswith(".netcdf"):
        try:
            import earthkit.data as ekd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "earthkit-data is required for netcdf support.\nTo install it, run `pip install earthkit-data`"
            )
        data = ekd.from_source("file", filename).to_xarray(mask_and_scale=False)
        if "Band1" in data:
            return from_d8(data["Band1"].values)
        elif "nextx" in data:
            x, y = data.nextx.values, data.nexty.values
            return from_cama_nextxy(x, y)
        elif "downx" in data:
            dx, dy = data.downx.values, data.downy.values
            return from_cama_downxy(dx, dy)
        else:
            raise OSError(f"Unknown filetype ({filename}).")
    else:
        raise OSError(f"Unknown filetype ({filename}).")


def load_cama_binfile(filename):
    f = open(filename.replace(".bin", ".ctl"), "r")
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
    return np.fromfile(filename, dtype=np.int32).reshape((nx, ny, 2), order="F")


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
