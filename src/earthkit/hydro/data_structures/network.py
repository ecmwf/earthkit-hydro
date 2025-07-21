import numpy as np
from earthkit.utils.array import to_device

from ._network import RiverNetworkStorage


class RiverNetwork:

    def __init__(self, river_network_storage: RiverNetworkStorage):
        self._storage = river_network_storage
        self.n_nodes = self._storage.n_nodes
        self.n_edges = self._storage.n_edges
        self.nodes = np.arange(self.n_nodes)
        self.sources = self._storage.sources
        self.sinks = self._storage.sinks
        # self.area = self._storage.area
        self.bifurcates = self._storage.bifurcates
        self._mask = self._storage.mask

        self.groups = np.split(self._storage.sorted_data, self._storage.splits, axis=1)

        del self._storage

    @property
    def mask(self):
        if self._mask is None:
            raise ValueError(
                "This RiverNetwork is not raster-based and does not have a mask."
            )
        return self._mask

    @property
    def shape(self):
        return self.mask.shape

    def __str__(self):
        return f"RiverNetwork with {self.n_nodes} nodes and {self.n_edges} edges."

    def __repr__(self):
        return self.__str__()

    def to_device(self, device, array_backend=None):
        self.groups = [to_device(group, device, array_backend) for group in self.groups]
        self._mask = to_device(self._mask, device, array_backend)
        # TODO: remove
        # self.nodes = to_device(self.nodes, device, array_backend)
        # self.sources = to_device(self.sources, device, array_backend)
        # self.sinks = to_device(self.sinks, device, array_backend)
        return self

    # def export(self, fpath="river_network.joblib", compression=1):
    #     joblib.dump(self._storage, fpath, compress=compression)

    def create_subnetwork(self, *args, **kwargs):
        raise NotImplementedError("Subnetwork creation not yet supported.")
