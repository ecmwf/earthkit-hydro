import networkx as nx


def test_create_from_edge_list(downstream_list):
    nodes = range(len(downstream_list))
    sinks, edges = [], []
    for nid, downstream in enumerate(downstream_list):
        if downstream == -1:
            sinks.append(nid)
        else:
            edges.append((nid, downstream))
    graph = nx.DiGraph(edges, sinks=sinks)

    assert graph.number_of_nodes() == len(nodes)
    assert graph.number_of_edges() == len(edges)
    assert set(graph.graph["sinks"]) == set(sinks)

    assert len(nx.dfs_tree(graph.reverse(), 16).edges) == len(nodes) - 1
    assert nx.dfs_tree(graph.reverse(), 7).number_of_nodes() == 3
