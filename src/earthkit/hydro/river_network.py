import earthkit.data
import numpy as np

from .graph import graph_manager


class RiverNetwork:
    def __init__(self, nodes, edges, graph_type) -> None:

        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.edges = edges
        graph_backend = graph_manager(graph_type)
        self.graph = graph_backend(nodes, edges)

        # downstream node (currently assume only exists one downstream node)
        self.downstream = np.ones(self.n_nodes, dtype=int)*-1
        for edge in self.edges:
            self.downstream[edge[1]] = edge[0]

        self.topologically_sorted = self.graph.topological_sorting()

        # upstream = [[]]*self.n_nodes
        # for edge in self.edges:
        #     upstream[edge[0]].append(edge[1])
        # self.upstream = upstream

    def accuflux(self, field):
        for node_index in self.topologically_sorted:
            if self.downstream[node_index] > 0:
                child_index = self.downstream[node_index]
                field[child_index] += field[node_index]
        return field
    
    # implementation using upstream nodes
    # def accuflux(self, field):
    #     for node_index in self.topologically_sorted:
    #         sum = field[node_index]
    #         for parent in self.upstream[node_index]:
    #             sum += field[parent]
    #         field[node_index] = sum
    #     return field


def from_d8(source, graph_type="igraph"):

    #  read river network from netcdf
    data = earthkit.data.from_source(**source).to_numpy()
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
    offsets = np.array(
        [0, ny - 1, ny, ny + 1, -1, 0, 1, -ny - 1, -ny, -ny + 1], dtype=int
    )
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
    return pd.DataFrame(
        {
            "ix": ix,
            "iy": iy,
            "lon": lon,
            "lat": lat,
            "downstream_id": downstream_nodes_indices,
        },
        index=nodes_indices)
