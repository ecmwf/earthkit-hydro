# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from ._rust import propagate_labels


def compute_topological_labels(
    sources: np.ndarray, sinks: np.ndarray, downstream_nodes: np.ndarray
):
    """Finds the topological distance labels for each node in
    downstream_nodes.

    Parameters
    ----------
    sources : numpy.ndarray
        The river network sources.
    sinks : numpy.ndarray
        The river network sinks.
    downstream_nodes : cnumpy.ndarray
        The river network downstream nodes.

    Returns
    -------
    numpy.ndarray
        Array of topological distance labels for each node.
    """

    n_nodes = np.uintp(downstream_nodes.shape[0])
    labels = np.zeros(n_nodes, dtype=np.int64)

    labels = propagate_labels(labels, sources, sinks, downstream_nodes, n_nodes)

    return labels
