import numpy as np

from ._rust import propagate_labels


def compute_topological_labels(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray
):

    n_nodes = np.uintp(downstream_nodes.shape[0])
    labels = np.zeros(n_nodes, dtype=np.int64)

    labels = propagate_labels(labels, sources, sinks, downstream_nodes, n_nodes)

    return labels
