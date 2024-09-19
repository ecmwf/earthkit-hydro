import time

import earthkit.data
import numpy as np
import xarray as xr

from .graph import graph_manager


class RiverNetwork:
    def __init__(self, nodes, edges, graph_type) -> None:
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.edges = edges
        graph_backend = graph_manager(graph_type)
        self.graph = graph_backend(self.n_nodes, edges)

        # downstream nodes (currently assume only exists one downstream node)
        self.downstream_nodes = np.ones(self.n_nodes, dtype=int)*self.n_nodes
        # self.downstream_nodes[self.downstream_nodes == -1] = None
        for edge in self.edges:
            self.downstream_nodes[edge[0]] = edge[1]

        self.topological_sort = self.graph.topological_sorting()
        self.topological_groups = self.construct_topological_groups()

        # upstream = [[]]*self.n_nodes
        # for edge in self.edges:
        #     upstream[edge[0]].append(edge[1])
        # self.upstream = upstream

    # def accuflux(self, field, in_place=True):
    #     if not in_place:
    #         field = field.copy()
    #     for node_index in self.topologically_sorted:
    #         if self.downstream_nodes[node_index] > 0:
    #             child_index = self.downstream_nodes[node_index]
    #             field[child_index] += field[node_index]
    #     return field

    # implementation using upstream nodes
    # def accuflux(self, field):
    #     for node_index in self.topologically_sorted:
    #         sum = field[node_index]
    #         for parent in self.upstream[node_index]:
    #             sum += field[parent]
    #         field[node_index] = sum
    #     return field

    # def find_sources(self):
    #     sources = self.graph.graph.vs.select(lambda v: v.indegree() == 0)
    #     self.sources = [x.index for x in sources]

    def get_inlets(self):
        tmp_nodes = self.nodes.copy()
        downstream_no_sinks = self.downstream_nodes[self.downstream_nodes != self.n_nodes]
        tmp_nodes[downstream_no_sinks] = -1
        inlets = tmp_nodes[tmp_nodes != -1]
        return inlets

    def topological_sort_numpy(self):

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
            print(labels)
        
        # labels[np.isinf(labels)] = n+1
        print(labels)
        groups = self.group_labels(labels)
        print(groups)

        return groups

    def group_labels(self, labels):
        # or could look at https://stackoverflow.com/questions/68331835/split-numpy-2d-array-based-on-separate-label-array
        sorted_indices = np.argsort(labels)  # sort by labels
        print("sorted indices")
        print(sorted_indices)
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = labels[sorted_indices]
        _, indices = np.unique(sorted_labels, return_index=True)
        subarrays = np.split(sorted_array, indices[1:])
        return subarrays

    # def construct_split_topological_sort(self):
    #     start_time = time.time()
    #     labels = np.zeros(self.n_nodes)

    #     stack = self.sources
    #     while len(stack)!=0:
    #         source = stack.pop()
    #         new_node = self.downstream_nodes[source]
    #         if new_node>0:
    #             candidate_label = labels[new_node]
    #             if labels[source] + 1 > candidate_label:
    #                 labels[new_node] = labels[source]+1
    #                 stack.append(new_node)
    #         else:
    #             labels[source] = np.inf
    #     start_time = time.time()
    #     self.split_topological_sort = self.split_array_by_labels(np.arange(self.n_nodes), labels)

    # def prepare_for_accuflux_splitter(self):
    #     self.find_sources()
    #     self.construct_split_topological_sort()

    def construct_topological_groups(self):
        labels = np.zeros(self.n_nodes)

        for node_index in self.topological_sort:
            if self.downstream_nodes[node_index] < self.n_nodes:
                child_index = self.downstream_nodes[node_index]
                if labels[node_index] + 1 > labels[child_index]:
                    labels[child_index] = labels[node_index] + 1
            else:
                labels[node_index] = np.inf
        split_topological_sort = self.group_labels(labels)
        return split_topological_sort

    def accuflux(self, field, in_place=True, operation=np.add):
        if not in_place:
            field = field.copy()
        topo_groups = self.topological_sort_numpy()
        for grouping in topo_groups[:-1]:  # exclude sinks here
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

    # def naive_splitter_accuflux(self, field):
    #     for grouping in self.topological_sort:
    #         for node_index in grouping[:-1]:
    #             if self.downstream_nodes[node_index] > 0:
    #                 child_index = self.downstream_nodes[node_index]
    #                 field[child_index] += field[node_index]
    #     return field


def from_netcdf_d8(filename, **kwargs):
    #  read river network from netcdf using xarray
    data = xr.open_dataset(filename, mask_and_scale=False)["Band1"].values
    return from_d8(data, **kwargs)


def from_d8(data, graph_type="igraph"):
    data_flat = data.flatten()

    # create mask to remove missing values and sinks
    mask_upstream = ((data != 255) & (data != 5)).flatten()
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
    nodes_matrix = np.ones(data_flat.shape, dtype=int) * -1
    nodes_matrix[mask_nodes] = nodes

    # create upstream and downstream nodes using local nodes indexing
    upstream_nodes = nodes_matrix[upstream_indices]
    downstream_nodes = nodes_matrix[downstream_indices]
    
    # create edges
    edges = list(zip(upstream_nodes, downstream_nodes))

    return RiverNetwork(nodes, edges, graph_type)


def from_camaflood(filename):
    downxy = xr.open_dataset(filename, mask_and_scale=False)

    dx, dy = downxy.downx.values, downxy.downy.values
    # return arrays of indices that are not ocean or sinks
    bmask = (dx != -999) & (dx != -9999)
    mask = np.where(bmask)
    # indicies of the non-ocean, non-sink cells
    indices = np.arange(dx.size).reshape(dx.shape)[mask].flatten()
    ncols = dx.shape[1]
    # calculate the indicies of the downstream cells
    ji = indices + dx[mask].flatten() * ncols + dy[mask].flatten()
    # translate 2d indices to 1d indices including sinks
    bmask = bmask.flatten()
    nodes = np.arange(np.sum(bmask), dtype=int)
    nodes_matrix = np.ones(bmask.shape, dtype=int) * -1
    nodes_matrix[bmask] = nodes
    nodes_indices = nodes_matrix[indices]
    downstream_nodes_indices = nodes_matrix[ji]
    # add ix, iy and lon, lat attributes
    ix, iy = np.meshgrid(np.arange(dx.shape[0]), np.arange(dx.shape[1]))
    ix, iy = ix.flatten()[indices], iy.flatten()[indices]
    lon, lat = np.meshgrid(downxy.lon.values, downxy.lat.values)
    lon, lat = lon.flatten()[indices], lat.flatten()[indices]

    edges = list(zip(nodes_indices, downstream_nodes_indices))
    nodes = np.arange(np.sum(dx != -999))

    return RiverNetwork(nodes, edges, graph_type)
