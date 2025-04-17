import numpy as np

from earthkit.hydro._rust import propagate_labels


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

    if inlets.shape[0] > 0:
        raise Exception("River Network contains a cycle.")

    labels[sinks] = n - 1  # put all sinks in last group in topological ordering

    return labels


def compute_topological_labels_rust(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray
):
    n_nodes = np.uintp(downstream_nodes.shape[0])
    inlets = downstream_nodes[sources]
    inlets = inlets[inlets != n_nodes]
    labels = np.zeros(n_nodes, dtype=np.int32)

    if (
        not labels.flags["C_CONTIGUOUS"]
        or not downstream_nodes.flags["C_CONTIGUOUS"]
        or not inlets.flags["C_CONTIGUOUS"]
    ):
        raise ValueError("Arrays must be contiguous in memory.")

    labels = propagate_labels(labels, inlets, downstream_nodes, n_nodes)

    labels[sinks] = n_nodes - 1  # put all sinks in last group in topological ordering

    return labels
