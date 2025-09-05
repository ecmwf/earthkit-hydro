import os

import numpy as np


def compute_topological_labels(sources, sinks, downstream_nodes, n_nodes):

    use_rust = int(os.environ.get("USE_RUST", "-1"))

    if use_rust == 0:
        func = compute_topological_labels_python
    elif use_rust == 1:
        from earthkit.hydro._rust import compute_topological_labels_rust as func
    else:
        try:
            from earthkit.hydro._rust import compute_topological_labels_rust as func
        except (ModuleNotFoundError, ImportError):
            func = compute_topological_labels_python

    return func(sources, sinks, downstream_nodes, n_nodes)


def compute_topological_labels_python(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray, n_nodes: int
):
    n_nodes = downstream_nodes.shape[0]
    inlets = downstream_nodes[sources]
    labels = np.zeros(n_nodes, dtype=np.intp)

    for n in range(1, n_nodes + 1):
        inlets = np.unique(inlets[inlets != n_nodes])  # subset to valid nodes
        if inlets.shape[0] == 0:
            break
        labels[inlets] = n  # update furthest distance from source
        inlets = downstream_nodes[inlets]

    if inlets.shape[0] != 0:
        raise ValueError("River Network contains a cycle.")
    labels[sinks] = n - 1  # put all sinks in last group in topological ordering

    return labels
