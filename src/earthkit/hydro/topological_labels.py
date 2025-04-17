import numpy as np


def compute_topological_labels_python(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray
):
    n_nodes = downstream_nodes.shape[0]
    inlets = downstream_nodes[sources]
    labels = np.zeros(n_nodes, dtype=int)

    for n in range(1, n_nodes + 1):
        inlets = inlets[inlets != n_nodes]  # subset to valid nodes
        if inlets.shape[0] == 0:
            break
        labels[inlets] = n  # update furthest distance from source
        inlets = downstream_nodes[inlets]

    if inlets.shape[0] != 0:
        raise ValueError("River Network contains a cycle.")

    labels[sinks] = n - 1  # put all sinks in last group in topological ordering

    return labels


def compute_topological_labels_rust(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray
):
    from ._rust import propagate_labels

    n_nodes = np.uintp(downstream_nodes.shape[0])
    labels = np.zeros(n_nodes, dtype=np.int64)

    labels = propagate_labels(labels, sources, sinks, downstream_nodes, n_nodes)

    return labels
