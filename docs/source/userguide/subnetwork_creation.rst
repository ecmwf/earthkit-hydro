Subnetwork creation
===================

By default, earthkit-hydro conducts operations over the full river network. In many applications, one is only interested in a specific subnetwork, such as a specific catchment or area.

There are two ways to create a subnetwork: masking nodes or masking edges.

Masking nodes
-------------

The simplest subnetwork creation mechanism is to remove nodes from a river network. This also removes any edges that are incoming or outgoing to any of the removed nodes.
The mask can be specified over the grid:

.. code-block:: python

    network = ekh.river_network.load("efas", "5")

    node_mask = np.ones(network.shape, dtype=bool)
    node_mask[10,10] = False

    subnetwork = ekh.subnetwork.from_mask(network, node_mask=node_mask)

Or as usual it is also possible to specify directly a mask on the nodes:

.. code-block:: python

    node_mask = np.ones(network.n_nodes, dtype=bool)
    node_mask[10] = False

    subnetwork = ekh.subnetwork.from_mask(network, node_mask=node_mask)


Masking edges
-------------

Masking edges is also possible. This is useful for controlling bifurcating river networks, or physically separating a subcatchment from the main catchment.

.. code-block:: python

    edge_mask = np.ones(network.n_edges, dtype=bool)
    edge_mask[10] = False

    subnetwork = ekh.subnetwork.from_mask(network, edge_mask=edge_mask)

Combining masks
---------------

It also possible to mask both nodes and edges in a single call.

.. code-block:: python

    node_mask = np.ones(network.n_nodes, dtype=bool)
    node_mask[10] = False

    edge_mask = np.ones(network.n_edges, dtype=bool)
    edge_mask[10] = False

    subnetwork = ekh.subnetwork.from_mask(network, node_mask=node_mask, edge_mask=edge_mask)
