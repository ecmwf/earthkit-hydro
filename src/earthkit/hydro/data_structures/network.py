import joblib
import numpy as np

from ._network import RiverNetworkStorage


class RiverNetwork:

    def __init__(self, river_network_storage: RiverNetworkStorage):
        self._storage = river_network_storage
        self.n_nodes = self._storage.n_nodes
        self.n_edges = self._storage.n_edges
        self.nodes = np.arange(self.n_nodes)
        self.sources = self._storage.sources
        self.sinks = self._storage.sinks
        self.area = self._storage.area
        self.bifurcates = self._storage.bifurcates

        self.groups = np.split(self._storage.sorted_data, self._storage.splits, axis=1)

    @property
    def mask(self):
        if self._storage.mask is None:
            raise ValueError(
                "This RiverNetwork is not raster-based and does not have a mask."
            )
        return self._storage.mask

    @property
    def shape(self):
        return self.mask.shape

    def __str__(self):
        return f"RiverNetwork with {self.n_nodes} nodes and {self.n_edges} edges."

    def __repr__(self):
        return self.__str__()

    def to(self, backend="numpy", dev=None):
        raise NotImplementedError(
            f"Switching array backend to {backend} on device {dev} not yet supported."
        )

    def export(self, fpath="river_network.joblib", compression=1):
        joblib.dump(self._storage, fpath, compress=compression)

    def create_subnetwork(self, *args, **kwargs):
        raise NotImplementedError("Subnetwork creation not yet supported.")
