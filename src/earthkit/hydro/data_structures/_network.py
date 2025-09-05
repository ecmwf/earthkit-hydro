import numpy as np
from earthkit.utils.array import to_device

from ._network_storage import RiverNetworkStorage


class RiverNetwork:
    """
    A class representing a river network for hydrological processing.

    Attributes
    ----------
    n_nodes : int
        The number of nodes in the river network.
    n_edges : int
        The number of nodes in the river network.
    sinks : array-like
        Nodes with no downstream connections.
    sources : array-like
        Nodes with no upstream connections.
    bifurcates : bool
        Whether the river network has bifurcations.
    shape : tuple
        The size of the river network grid. None if the network is vector-based.
    mask : array-like
        Flattened 1D indices on the raster grid corresponding to river network nodes.
    array_backend : str
        The array backend of the river network.
    device : str
        The device of the river network.
    return_type : str
        The default return type of the river network. Either "gridded" or "masked".
    """

    def __init__(self, river_network_storage: RiverNetworkStorage):
        self._storage = river_network_storage
        self.n_nodes = self._storage.n_nodes
        self.n_edges = self._storage.n_edges
        self.sources = self._storage.sources
        self.sinks = self._storage.sinks

        self.bifurcates = self._storage.bifurcates
        self.edge_weights = self._storage.edge_weights

        self.mask = self._storage.mask
        self.shape = self._storage.shape
        self.array_backend = "numpy"
        self.device = "cpu"
        self.return_type = "gridded"

        self.coords = self._storage.coords

        self.data = [self._storage.sorted_data]
        self.groups = np.split(self._storage.sorted_data, self._storage.splits, axis=1)

    def __str__(self):
        return f"RiverNetwork with {self.n_nodes} nodes and {self.n_edges} edges."

    def __repr__(self):
        return self.__str__()

    def to_device(self, device=None, array_backend=None):
        """
        Change the RiverNetwork's array backend and/or move it to a
        different device.

        Parameters
        ----------
        device : str, optional
            The device to which to transfer. Default is None, which is `'cpu'` for all backends except cupy, which is `'gpu'`.
        array_backend : str, optional
            The array backend.
            One of "numpy", "np", "cupy", "cp", "pytorch", "torch", "jax", "jnp", "tensorflow" or "tf".
            Default is None, which uses `self.array_backend`.

        Returns
        -------
        RiverNetwork
            The modified RiverNetwork.
        """

        # TODO: use xp.asarray
        if array_backend == "np":
            array_backend = "numpy"
        elif array_backend == "cp":
            array_backend = "cupy"
        elif array_backend == "jnp":
            array_backend = "jax"
        elif array_backend == "tf":
            array_backend = "tensorflow"
        elif array_backend == "pytorch":
            array_backend = "torch"

        if device is None:
            device = "cpu" if array_backend != "cupy" else "gpu"
        if array_backend is None:
            if self.array_backend == "numpy" and device in ["gpu", "cuda"]:
                array_backend = "cupy"
            else:
                array_backend = self.array_backend

        if array_backend in ["torch", "cupy", "numpy"]:
            self.groups = [
                to_device(group, device, array_backend=array_backend)
                for group in self.groups
            ]
            self.mask = to_device(self.mask, device, array_backend=array_backend)
            self.data = [to_device(self.data[0], device, array_backend=array_backend)]
        elif array_backend == "jax":
            assert device == "cpu"
            import jax.numpy as jnp

            self.groups = [jnp.array(x) for x in self.groups]
            self.mask = jnp.array(self.mask)
            self.data = [jnp.array(self.data[0])]
        elif array_backend == "tensorflow":
            assert device == "cpu"
            import tensorflow as tf

            self.groups = [tf.convert_to_tensor(x, dtype=tf.int32) for x in self.groups]
            self.mask = tf.convert_to_tensor(self.mask, dtype=tf.int32)
            self.data = [tf.convert_to_tensor(self.data[0], dtype=tf.int32)]
        else:
            raise NotImplementedError

        self.array_backend = array_backend
        self.device = device
        return self

    def set_default_return_type(self, return_type):
        """
        Set the default return type for the river network.

        Parameters
        ----------
        return_type : str
            The default return_type to use.

        Returns
        -------
        None
        """
        if return_type not in ["gridded", "masked"]:
            raise ValueError(
                f'Invalid return_type {return_type}. Valid types are "gridded", "masked"'
            )
        self.return_type = return_type

    def export(self, fpath="river_network.joblib", compression=1):
        """
        Save the river network to a local file.

        Parameters
        ----------
        fpath : str, optional
            The filepath specifying where to save the RiverNetwork. Default is `'river_network.joblib'`.
        compression : str, optional
            The compression factor used for saving. Default is 1.

        Returns
        -------
        None
        """
        import joblib

        joblib.dump(self._storage, fpath, compress=compression)
