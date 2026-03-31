Catchment statistics
====================

A very common hydrological task is computing statistics over river basins. This is very simple in earthkit-hydro.

Calculation
-----------

A catchment of a gauge location is defined as all nodes flowing to that location.
Catchment statistics are calculated for each location in the same manner as for upstream statistics, with optional weights.
The only difference is that for catchment statistics, one specifies directly the gauge locations of interest as opposed to computing for each node in the river network.
The following methods are available:

.. code-block:: python

    network = ekh.river_network.load("efas", "5")
    field = np.ones(network.n_nodes)
    node_weights = np.ones(network.n_nodes)  # optional weights for the nodes
    edge_weights = np.ones(network.n_edges)  # optional weights for the edges
    locations = {
        "station1": (10, 10),
        "station2": (20, 20)
    }

    upstream_sum = ekh.catchments.sum(network, field, locations, node_weights, edge_weights)
    upstream_mean = ekh.catchments.mean(network, field, locations, node_weights, edge_weights)
    upstream_max = ekh.catchments.max(network, field, locations, node_weights, edge_weights)
    upstream_min = ekh.catchments.min(network, field, locations, node_weights, edge_weights)
    upstream_std = ekh.catchments.std(network, field, locations, node_weights, edge_weights)
    upstream_var = ekh.catchments.var(network, field, locations, node_weights, edge_weights)
