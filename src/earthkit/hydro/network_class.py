import joblib
import numpy as np

from .utils import mask_2d


class RiverNetwork:
    """A class representing a river network for hydrological processing.

    Attributes
    ----------
    nodes : numpy.ndarray
        Array containing the node ids of the river network.
    n_nodes : int
        The number of nodes in the river network.
    downstream_nodes : numpy.ndarray
        Array of downstream node ids corresponding to each node.
    mask : numpy.ndarray
        A boolean mask converting from the domain to the river network.
    sinks : numpy.ndarray
        Nodes with no downstream connections.
    sources : numpy.ndarray
        Nodes with no upstream connections.
    topological_labels : numpy.ndarray
        A topological group label for each node.
    topological_groups : list of numpy.ndarray
        Groups of nodes sorted in topological order.

    """

    def __init__(
        self,
        nodes,
        downstream,
        mask,
        sinks=None,
        sources=None,
        topological_labels=None,
        check_for_cycles=False,
    ) -> None:
        """Initialises the RiverNetwork with nodes, downstream nodes, and a
        mask.

        Parameters
        ----------
        nodes : numpy.ndarray
            Array containing the node ids of the river network.
        downstream : numpy.ndarray
            Array of downstream node ids corresponding to each node.
        mask : numpy.ndarray
            A mask converting from the domain to the river network.
        sinks : numpy.ndarray, optional
            Array of sinks of the river network.
        sources :  numpy.ndarray, optional
            Array of sources of the river network.
        topological_labels : numpy.ndarray, optional
            Array of precomputed topological distance labels.
        check_for_cycles : bool, optional
            Whether to check for cycles when instantiating the river network.

        """
        self.nodes = nodes
        self.n_nodes = len(nodes)
        self.downstream_nodes = downstream
        self.mask = mask
        self.sinks = (
            sinks
            if sinks is not None
            else self.nodes[self.downstream_nodes == self.n_nodes]
        )  # nodes with no downstreams
        self.sources = (
            sources if sources is not None else self.get_sources()
        )  # nodes with no upstreams
        if check_for_cycles:
            self.check_for_cycles()
        self.topological_labels = (
            topological_labels
            if topological_labels is not None
            else self.compute_topological_labels()
        )
        self.topological_groups = self.topological_groups_from_labels()

    @property
    def shape(self):
        return self.mask.shape

    def get_sources(self):
        """Identifies the source nodes in the river network (nodes with no
        upstream nodes).

        Returns
        -------
        numpy.ndarray
            Array of source nodes.

        """
        tmp_nodes = self.nodes.copy()
        downstream_no_sinks = self.downstream_nodes[
            self.downstream_nodes != self.n_nodes
        ]  # get all downstream nodes
        tmp_nodes[downstream_no_sinks] = -1  # downstream nodes that aren't sinks = -1
        inlets = tmp_nodes[
            tmp_nodes != -1
        ]  # sources are nodes that are not downstream nodes
        return inlets

    def check_for_cycles(self):
        """Checks if the river network contains any cycles and raises an
        Exception if it does.
        """
        nodes = self.downstream_nodes.copy()
        while True:
            if np.any(nodes == self.nodes):
                Exception("River Network contains a cycle.")
            elif np.all(nodes == self.n_nodes):
                break
            not_sinks = nodes != self.n_nodes
            nodes[not_sinks] = self.downstream_nodes[nodes[not_sinks]].copy()

    def compute_topological_labels(self):
        """Finds the topological distance labels for each node in the river
        network.

        Returns
        -------
        numpy.ndarray
            Array of topological distance labels for each node.

        """
        inlets = self.sources
        labels = np.zeros(self.n_nodes, dtype=int)
        old_sum = -1
        current_sum = 0  # sum of labels
        n = 1  # distance from source
        while current_sum > old_sum:
            if n > self.n_nodes:
                raise Exception("River Network contains a cycle.")
            old_sum = current_sum
            inlets = inlets[inlets != self.n_nodes]  # subset to valid nodes
            labels[inlets] = n  # update furthest distance from source
            inlets = self.downstream_nodes[inlets]
            n += 1
            current_sum = np.sum(labels)
        labels[self.sinks] = n  # put all sinks in last group in topological ordering
        return labels

    def topological_groups_from_labels(self):
        """Groups nodes by their topological distance labels.

        Parameters
        ----------
        labels : numpy.ndarray
            Array of labels representing the topological distances of nodes.

        Returns
        -------
        list of numpy.ndarray
            A list of subarrays, each containing nodes with the same label.

        """
        sorted_indices = np.argsort(self.topological_labels)  # sort by labels
        sorted_array = self.nodes[sorted_indices]
        sorted_labels = self.topological_labels[sorted_indices]
        _, indices = np.unique(
            sorted_labels, return_index=True
        )  # find index of first occurrence of each label
        subarrays = np.split(
            sorted_array, indices[1:]
        )  # split array at each first occurrence of a label
        return subarrays

    def export(self, fpath="river_network.joblib", compression=1):
        """Exports the river network instance to a file.

        Parameters
        ----------
        fpath : str, optional
            The filepath to save the instance (default is "river_network.joblib").
        compression : int, optional
            Compression level for joblib (default is 1).

        """
        joblib.dump(self, fpath, compress=compression)

    @mask_2d
    def create_subnetwork(self, field, recompute=False):
        """Creates a subnetwork from the river network based on a mask.

        Parameters
        ----------
        field : numpy.ndarray
            A boolean mask to subset the river network.
        recompute : bool, optional
            If True, recomputes the topological labels for the subnetwork.
            Default is False.

        Returns
        -------
        RiverNetwork
            A subnetwork of the river network.

        """
        river_network_mask = field
        valid_indices = np.where(self.mask)
        new_valid_indices = (
            valid_indices[0][river_network_mask],
            valid_indices[1][river_network_mask],
        )
        domain_mask = np.full(self.mask.shape, False)
        domain_mask[new_valid_indices] = True

        downstream_indices = self.downstream_nodes[river_network_mask]
        n_nodes = len(downstream_indices)  # number of nodes in the subnetwork
        # create new array of network nodes, setting all nodes not in mask to n_nodes
        subnetwork_nodes = np.full(self.n_nodes, n_nodes)
        subnetwork_nodes[river_network_mask] = np.arange(n_nodes)
        # get downstream nodes in the subnetwork
        non_sinks = np.where(downstream_indices != self.n_nodes)
        downstream = np.full(n_nodes, n_nodes)
        downstream[non_sinks] = subnetwork_nodes[downstream_indices[non_sinks]]
        nodes = np.arange(n_nodes)

        if not recompute:
            sinks = nodes[downstream == n_nodes]
            topological_labels = self.topological_labels[river_network_mask]
            topological_labels[sinks] = self.n_nodes

            return RiverNetwork(
                nodes,
                downstream,
                domain_mask,
                sinks=sinks,
                topological_labels=topological_labels,
            )
        else:
            return RiverNetwork(nodes, downstream, domain_mask)
