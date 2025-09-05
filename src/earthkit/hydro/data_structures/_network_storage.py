class RiverNetworkStorage:
    def __init__(
        self,
        n_nodes,
        n_edges,
        sorted_data,  # np.vstack((down_ids_upsort, up_ids_upsort, edge_ids_upsort))
        sources,
        sinks,
        coords,
        splits,  # indices of where to split sorted_data
        area,
        mask,
        shape,
        bifurcates=False,
        edge_weights=None,
    ):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.bifurcates = bifurcates
        self.sources = sources
        self.sinks = sinks
        self.coords = coords
        self.area = area
        self.sorted_data = sorted_data
        self.splits = splits
        self.mask = mask
        self.shape = shape
        self.edge_weights = edge_weights
        assert not (bifurcates and edge_weights is None)
