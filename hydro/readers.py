import numpy as np
import xarray as xr
from .river_network import RiverNetwork

def from_netcdf_d8(filename, **kwargs):
    #  read river network from netcdf using xarray
    data = xr.open_dataset(filename, mask_and_scale=False)["Band1"].values
    return from_d8(data, **kwargs)

def from_d8(data):
    data_flat = data.flatten()

    # create mask to remove missing values and sinks
    mask_upstream = (data_flat != 255) & (data_flat != 5)
    directions = data_flat[mask_upstream].astype("int")

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
    y_offsets = np.array([0, -1, -1, -1, 0, 0, 0, 1, 1, 1])[directions]
    nx, ny = data.shape
    missing_mask = data_flat != 255

    return create_network(x_offsets, y_offsets, missing_mask, mask_upstream, nx, ny)

def from_netcdf_cama(filename, **kwargs):
    data = xr.open_dataset(filename, mask_and_scale=False)
    return from_cama(data)

def from_cama(data):
    x_offsets = data.downx.values
    nx, ny = x_offsets.shape
    x_offsets = x_offsets.flatten()
    mask_upstream = ((x_offsets != -999) & (x_offsets != -9999)) & (x_offsets != -1000)
    x_offsets = x_offsets[mask_upstream]
    y_offsets = data.downy.values.flatten()[mask_upstream]
    missing_mask = x_offsets != -9999

    return create_network(x_offsets, y_offsets, missing_mask, mask_upstream, nx, ny)

def create_network(x_offsets, y_offsets, missing_mask, sink_missing_mask, nx, ny):
    upstream_indices = np.arange(missing_mask.size)[sink_missing_mask] # all indices, including missing values

    x_coords = upstream_indices % ny # old x_coords
    x_coords = (x_coords + x_offsets) % ny # new x_coords

    y_coords = np.floor_divide(upstream_indices, ny) # old y_coords
    y_coords = (y_coords + y_offsets) % nx # new y_coords
    
    downstream_indices = x_coords + y_coords * ny

    # relabel to only index nodes that are not missing
    n_nodes = np.sum(missing_mask)
    nodes = np.arange(n_nodes, dtype=int)

    # put back nodes indices in original 1d array
    nodes_matrix = np.ones(missing_mask.size, dtype=int) * n_nodes
    nodes_matrix[missing_mask] = nodes

    # create upstream and downstream nodes using local nodes indexing
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    
    # downstream nodes (currently assume only exists one downstream node)
    downstream = np.ones(n_nodes, dtype=int) * n_nodes
    
    downstream[upstream_nodes] = downstream_nodes

    return RiverNetwork(nodes, downstream)
