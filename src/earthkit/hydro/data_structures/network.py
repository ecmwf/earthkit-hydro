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
        self.mask = self._storage.mask
        self.shape = self._storage.shape

        self.groups = np.split(self._storage.sorted_data, self._storage.splits, axis=1)

    def __str__(self):
        return f"RiverNetwork with {self.n_nodes} nodes and {self.n_edges} edges."

    def __repr__(self):
        return self.__str__()

    def to_device(self, device=None, array_backend=None):

        # shorthands
        if array_backend == "np":
            array_backend = "numpy"
        elif array_backend == "cp":
            array_backend = "cupy"
        elif array_backend == "jnp":
            array_backend = "jax"
        elif array_backend == "tf":
            array_backend = "tensorflow"

        if device is None:
            device = "cpu" if array_backend != "cupy" else "gpu"

        if array_backend in ["torch", "cupy", "numpy"]:
            self.groups = [
                to_device(group, device, array_backend=array_backend)
                for group in self.groups
            ]
            self.mask = to_device(self.mask, device, array_backend=array_backend)
        elif array_backend == "jax":
            assert device == "cpu"
            import jax.numpy as jnp

            self.groups = [jnp.array(x) for x in self.groups]
            self.mask = jnp.array(self.mask)
        elif array_backend == "tensorflow":
            assert device == "cpu"
            import tensorflow as tf

            self.groups = [tf.convert_to_tensor(x, dtype=tf.int32) for x in self.groups]
            self.mask = tf.convert_to_tensor(self.mask, dtype=tf.int32)
        else:
            raise NotImplementedError
        return self

    def export(self, fpath="river_network.joblib", compression=1):
        import joblib

        joblib.dump(self._storage, fpath, compress=compression)

    def create_subnetwork(self, *args, **kwargs):
        raise NotImplementedError("Subnetwork creation not yet supported.")
