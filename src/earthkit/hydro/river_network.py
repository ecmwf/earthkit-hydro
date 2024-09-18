import earthkit.data
import numpy as np

from .graph import graph_manager


class RiverNetwork:
    def __init__(self, nodes, edges, graph_type) -> None:

        self.nodes = nodes
        self.edges = edges
        graph_backend = graph_manager(graph_type)
        self.graph = graph_backend(nodes, edges)


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
