Flow accumulations
==================

Flow accumulations are a fundamental aspect of hydrological modeling, allowing for the analysis of how water flows through a river network.
Fundamentally, there are two different types of flow accumulations: full flow accumulations (global aggregation) and one-step neighbor accumulations (local aggregation).

Full flow accumulation (global aggregation)
-------------------------------------------

A full flow accumulation is a global aggregation of flow across the entire river network.

.. image:: ../../images/accuflux.gif
   :width: 250px
   :align: right

Typically, this is computed by starting from the sources and flowing downstream until the sinks.
The most common aggregation function is the sum, but it also possible to compute averages, maximums, minimums, etc. over all upstream nodes.

This can be done in earthkit-hydro using the `upstream` submodule, which computes a metric of a field over all upstream nodes in the river network.

.. raw:: html

   <br style="clear: both">

.. code-block:: python

    network = ekh.river_network.load("efas", "5")
    field = np.ones(network.n_nodes)
    node_weights = np.ones(network.n_nodes)  # optional weights for the nodes
    edge_weights = np.ones(network.n_edges)  # optional weights for the edges

    upstream_sum = ekh.upstream.sum(network, field, node_weights, edge_weights)
    upstream_mean = ekh.upstream.mean(network, field, node_weights, edge_weights)
    upstream_max = ekh.upstream.max(network, field, node_weights, edge_weights)
    upstream_min = ekh.upstream.min(network, field, node_weights, edge_weights)
    upstream_std = ekh.upstream.std(network, field, node_weights, edge_weights)
    upstream_var = ekh.upstream.var(network, field, node_weights, edge_weights)

Whilst typically flow accumulations go from sources to sinks, it is also possible to compute the flow accumulation in the reverse direction, from sinks to sources.
The `downstream` submodule provides this functionality, with an analagous API to the `upstream` submodule.

.. code-block:: python

    downstream_sum = ekh.downstream.sum(network, field, node_weights, edge_weights)
    downstream_mean = ekh.downstream.mean(network, field, node_weights, edge_weights)
    downstream_max = ekh.downstream.max(network, field, node_weights, edge_weights)
    downstream_min = ekh.downstream.min(network, field, node_weights, edge_weights)
    downstream_std = ekh.downstream.std(network, field, node_weights, edge_weights)
    downstream_var = ekh.downstream.var(network, field, node_weights, edge_weights)

One-step neighbor accumulation (local aggregation)
--------------------------------------------------

Contrarily to a global accumulation, a one-step neighbor accumulation is a local aggregation of flow across the immediate neighbors of each node in the river network.
This is analagous to a message passing operation in graph networks, where each node receives a message from its neighbors and aggregates it.

Again, typically this is computed downstream, but it is also possible to compute it upstream. Both of these functionalities are provided by the `move` submodule.
The aggregation function is specified via the `metric` argument.

.. code-block:: python

    ekh.move.downstream(network, field, node_weights, edge_weights, metric='sum')
    ekh.move.upstream(network, field, node_weights, edge_weights, metric='sum')
