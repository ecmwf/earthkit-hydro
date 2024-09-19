import igraph as ig
import numpy as np


def graph_manager(graph_type):
    if graph_type == "igraph":
        return IGraph
    else:
        raise NotImplementedError


class Graph:
    def topological_sorting():
        return NotImplementedError


class IGraph(Graph):
    def __init__(self, nodes, edges):
        self.graph = ig.Graph(n=nodes, edges=edges, directed=True)

    def topological_sorting(self):
        return np.array(self.graph.topological_sorting())

    def subgraph(self, nodes):
        subgraph_nodes = self.graph.neighborhood(nodes, order=len(self.graph.vs), mode='out')
        new_graph = self.graph.subgraph(subgraph_nodes)
        print(new_graph.summary())
        return new_graph
