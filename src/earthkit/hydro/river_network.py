import numpy as np
import xarray as xr

class RiverNetwork:
    def __init__(self, nodes, downstream, graph_type) -> None:
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.downstream_nodes = downstream
        self.topological_groups = self.topological_sort()

    def get_inlets(self):
        tmp_nodes = self.nodes.copy()
        downstream_no_sinks = self.downstream_nodes[self.downstream_nodes != self.n_nodes]
        tmp_nodes[downstream_no_sinks] = -1
        inlets = tmp_nodes[tmp_nodes != -1]
        return inlets

    def topological_sort(self):
        inlets = self.get_inlets()
        labels = np.zeros(self.n_nodes)
        n = 1
        while np.any(labels == 0):
            inlets = inlets[inlets != self.n_nodes]
            labels[inlets] = np.maximum(labels[inlets], n)
            mask_sinks = self.downstream_nodes[inlets] == self.n_nodes
            labels[self.nodes[inlets][mask_sinks]] = np.inf
            
            inlets = self.downstream_nodes[inlets]
            n += 1
        groups = self.group_labels(labels)

        return groups

    def group_labels(self, labels):
        # or could look at https://stackoverflow.com/questions/68331835/split-numpy-2d-array-based-on-separate-label-array
        sorted_indices = np.argsort(labels)  # sort by labels
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = labels[sorted_indices]
        _, indices = np.unique(sorted_labels, return_index=True)
        subarrays = np.split(sorted_array, indices[1:])
        return subarrays

    def accuflux(self, field, in_place=True, operation=np.add):
        if not in_place:
            field = field.copy()
        for grouping in self.topological_groups[:-1]:  # exclude sinks here
            operation.at(field, self.downstream_nodes[grouping], field[grouping])
        return field

    def upstream_points(self):
        ones = np.ones(self.n_nodes)
        return self.accuflux(ones)

    def upstream(self, field):
        mask = self.downstream_nodes != self.n_nodes
        ups = np.zeros(self.n_nodes, dtype=field.dtype)
        np.add.at(ups, self.downstream_nodes[mask], field[mask])
        return ups

    def downstream(self, field):
        down = np.zeros(self.n_nodes, dtype=field.dtype)
        mask = self.downstream_nodes != self.n_nodes
        down[mask] = field[self.downstream_nodes[mask]]
        return down

    def catchment(self, nodes):
        catchments = self.graph.subgraph(nodes)


def from_netcdf_d8(filename, **kwargs):
    #  read river network from netcdf using xarray
    data = xr.open_dataset(filename, mask_and_scale=False)["Band1"].values
    return from_d8(data, **kwargs)


def from_d8(data, graph_type="igraph"):
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
    mask_nodes = data_flat != 255

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

    return RiverNetwork(nodes, downstream, graph_type)


def from_camaflood(filename, graph_type="igraph"):
    downxy = xr.open_dataset(filename, mask_and_scale=False)
    dx = downxy.downx.values
    nx = dx.shape[0]
    ny = dx.shape[1]
    dx_flat = dx.flatten()
    del dx
    # , dy_flat = downxy.downx.values.flatten(), downxy.downy.values.flatten()

    mask_upstream = ((dx_flat != -999) & (dx_flat != -9999)) & (dx_flat != -1000) # 1d flattened indices

    upstream_indices = np.arange(dx_flat.size)[mask_upstream] # indices with names according to old 2d array
    
    x_coords = upstream_indices % ny
    new_x_coords = (x_coords + dx_flat[mask_upstream]) % ny
    del x_coords
    del dx_flat

    dy_flat = downxy.downy.values.flatten()
    y_coords = np.floor_divide(upstream_indices, ny)
    new_y_coords = (y_coords + dy_flat[mask_upstream]) % nx
    del y_coords

    downstream_indices = new_x_coords + new_y_coords * ny
    del new_x_coords
    del new_y_coords

    # nodes mask, only removing missing values (oceans)
    mask_nodes = dy_flat != -9999
    del dy_flat

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

    return RiverNetwork(nodes, downstream, graph_type)
