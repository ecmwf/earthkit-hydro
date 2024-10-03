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
    ny = data.shape[1]
    offsets = np.array([0, ny - 1, ny, ny + 1, -1, 0, 1, -ny - 1, -ny, -ny + 1], dtype=int)
    downstream_offset = offsets[directions]

    # upstream indices
    upstream_indices = np.arange(data.size)[mask_upstream]
    # downstream indices
    downstream_indices = upstream_indices + downstream_offset

    # nodes mask, only removing missing values (oceans)
    mask = data != 255
    mask_nodes = mask.flatten()

    # create simple local indexing for nodes, 0 to n
    n_nodes = np.sum(mask_nodes)
    nodes = np.arange(n_nodes, dtype=int)

    # put back nodes indices in original 1d array
    nodes_matrix = np.ones(data_flat.shape, dtype=int) * n_nodes
    nodes_matrix[mask_nodes] = nodes

    # create upstream and downstream nodes using local nodes indexing
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    
    # downstream nodes (currently assume only exists one downstream node)
    downstream = np.ones(n_nodes, dtype=int)*n_nodes
    
    downstream[upstream_nodes] = downstream_nodes

    return RiverNetwork(nodes, downstream, mask)


def from_camaflood(filename):
    downxy = xr.open_dataset(filename, mask_and_scale=False)
    dy_flat = downxy.downy.values.flatten()
    dx = downxy.downx.values
    del downxy
    nx = dx.shape[0]
    ny = dx.shape[1]
    dx_flat = dx.flatten()
    mask = dx != -9999
    del dx

    mask_upstream = ((dx_flat != -999) & (dx_flat != -9999)) & (dx_flat != -1000) # 1d flattened indices

    upstream_indices = np.arange(dx_flat.size)[mask_upstream] # indices with names according to old 2d array
    
    x_coords = upstream_indices % ny
    new_x_coords = (x_coords + dx_flat[mask_upstream]) % ny
    del x_coords
    del dx_flat

    y_coords = np.floor_divide(upstream_indices, ny)
    new_y_coords = (y_coords + dy_flat[mask_upstream]) % nx
    del y_coords
    del dy_flat

    downstream_indices = new_x_coords + new_y_coords * ny
    del new_x_coords
    del new_y_coords

    # nodes mask, only removing missing values (oceans)
    mask_nodes = mask.flatten()

    # create simple local indexing for nodes, 0 to n
    n_nodes = np.sum(mask_nodes)
    nodes = np.arange(n_nodes, dtype=int)

    # put back nodes indices in original 1d array
    nodes_matrix = np.ones(nx*ny, dtype=int) * n_nodes
    nodes_matrix[mask_nodes] = nodes
    del mask_nodes

    # create upstream and downstream nodes using local nodes indexing
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    del nodes_matrix

    # downstream nodes (currently assume only exists one downstream node)
    downstream = np.ones(n_nodes, dtype=int) * n_nodes
    
    downstream[upstream_nodes] = downstream_nodes
    del upstream_nodes
    del downstream_nodes

    return RiverNetwork(nodes, downstream, mask)
