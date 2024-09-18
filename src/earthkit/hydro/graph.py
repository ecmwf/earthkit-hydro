import igraph as ig


def graph_manager(graph_type):
    if graph_type == "igraph":
        return IGraph
    else:
        raise NotImplementedError


class Graph:
    def something():
        return NotImplementedError


class IGraph(Graph):
    def __init__(self, nodes, edges):
        self.graph = ig.Graph(n=nodes, edges=edges, directed=True)
