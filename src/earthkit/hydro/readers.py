import numpy as np
import xarray as xr
from .river_network import RiverNetwork
import joblib
from .caching import Cache
import tempfile


def load_river_network(
    domain="efas",
    version="5",
    cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
    data_source="https://github.com/Oisin-M/river_network_store/raw/refs/heads/develop/{ekh_version}/{domain}/{version}/river_network.joblib",
    cache_fname="{ekh_version}_{domain}_{version}.joblib",
):
    from ._version import __version__ as ekh_version

    cache = Cache(cache_dir, data_source, cache_fname)
    filepath = cache(ekh_version=ekh_version[0:3], domain=domain, version=version)
    network = joblib.load(filepath)

    return network


def from_netcdf_d8(filename):
    # read river network from netcdf using xarray
    data = xr.open_dataset(filename, mask_and_scale=False)["Band1"].values

    return from_d8(data)


def from_d8(data):
    shape = data.shape
    data_flat = data.flatten()
    del data

    # create mask to remove missing values and sinks
    mask_upstream = (data_flat != 255) & (data_flat != 5)
    missing_mask = data_flat != 255
    directions = data_flat[mask_upstream].astype("int")
    del data_flat

    # create offsets from d8 convention (numpad directions):
    # 9 +1 +1
    # 8  0 +1
    # 7 -1 +1
    # 6 +1  0
    # 5  0  0 (sink)
    # 4 -1  0
    # 3 +1 -1
    # 2  0 -1
    # 1 -1 -1
    x_offsets = np.array([0, -1, 0, +1, -1, 0, +1, -1, 0, +1])[directions]
    y_offsets = -np.array([0, -1, -1, -1, 0, 0, 0, 1, 1, 1])[directions]
    del directions

    upstream_indices, downstream_indices = find_upstream_downstream_indices_from_offsets(
        x_offsets, y_offsets, missing_mask, mask_upstream, shape
    )

    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def from_netcdf_cama(filename, type="downxy"):
    data = xr.open_dataset(filename, mask_and_scale=False)
    if type == "downxy":
        dx, dy = data.downx.values, data.downy.values
        return from_cama_downxy(dx, -dy)
    elif type == "nextxy":
        x, y = data.nextx.values, data.nexty.values
        return from_cama_nextxy(x, y)
    else:
        raise Exception("Unknown type")


def from_bin_cama(filename, type="downxy"):
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
    x_offsets = dx
    y_offsets = dy.flatten()
    shape = x_offsets.shape
    x_offsets = x_offsets.flatten()
    mask_upstream = ((x_offsets != -999) & (x_offsets != -9999)) & (x_offsets != -1000)
    missing_mask = x_offsets != -9999
    x_offsets = x_offsets[mask_upstream]
    y_offsets = y_offsets[mask_upstream]

    upstream_indicies, downstream_indices = find_upstream_downstream_indices_from_offsets(
        x_offsets, y_offsets, missing_mask, mask_upstream, shape
    )

    return create_network(upstream_indicies, downstream_indices, missing_mask, shape)


def from_cama_nextxy(x, y):
    shape = x.shape
    x = x.flatten()
    missing_mask = x != -9999
    mask_upstream = ((x != -9) & (x != -9999)) & (x != -10)
    upstream_indices = np.arange(x.size)[mask_upstream]
    x = x[mask_upstream]
    y = y.flatten()[mask_upstream]

    # Fortran counts from 1, we do not
    x -= 1
    y -= 1

    downstream_indices = x + y * shape[1]

    return create_network(upstream_indices, downstream_indices, missing_mask, shape)


def find_upstream_downstream_indices_from_offsets(x_offsets, y_offsets, missing_mask, mask_upstream, shape):
    ny, nx = shape
    upstream_indices = np.arange(missing_mask.size)[mask_upstream]  # all indices, including missing values
    del mask_upstream

    x_coords = upstream_indices % nx
    x_coords = (x_coords + x_offsets) % nx
    downstream_indices = x_coords
    del x_coords

    y_coords = np.floor_divide(upstream_indices, nx)  # old y_coords
    y_coords = (y_coords + y_offsets) % ny  # new y_coords

    downstream_indices += y_coords * nx
    del y_coords

    return upstream_indices, downstream_indices


def create_network(upstream_indices, downstream_indices, missing_mask, shape):
    # relabel to only index nodes that are not missing
    n_nodes = int(np.sum(missing_mask))
    nodes = np.arange(n_nodes, dtype=int)

    # put back nodes indices in original 1d array
    nodes_matrix = np.ones(missing_mask.size, dtype=int) * n_nodes
    nodes_matrix[missing_mask] = nodes

    # create upstream and downstream nodes using local nodes indexing
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    del upstream_indices, downstream_indices, nodes_matrix

    # downstream nodes (currently assume only exists one downstream node)
    downstream = np.ones(n_nodes, dtype=int) * n_nodes

    downstream[upstream_nodes] = downstream_nodes

    print("data loaded, initialising river network")

    return RiverNetwork(nodes, downstream, missing_mask.reshape(shape))
