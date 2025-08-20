Distance and length calculations
================================

In earthkit-hydro, a distinction is made between distance and length calculations. This is necessary because fundamentally the two operations are not equivalent, although they are often conflated.

Distinction between a length and distance
-----------------------------------------

In essence, lengths and distances try and capture two different quantities based on two different inputs.
A length is calculated by considering the length of a river network in each gridcell or graph node. As a result, lengths are *node properties*.
There is only one length per gridcell, even if a confluence or bifurcation occurs.

Distances on the other hand are not measured at a node, but are rather specified in terms of the distance from one gridcell to another.
As such, they are an *edge property* and distances can be different for each branch at a confluence and a bifurcation.

Even in simple river networks without confluences or bifurcations, lengths and distances are still not equivalent. This distinction is clear in the following contrived example.

.. image:: ../../images/distance_length.png
   :width: 500px
   :align: center

The length here for the highlighted segment is 3.

By contrast, the distance here for the highlighted segment is only 2.

Maximum and minimum distances or lengths
----------------------------------------

In the above example, there was only a single path from the source node to the terminal node.
However, in river networks there will often by many paths. It is thus important to consider whether one is interested in the shortest or longest path.

In earthkit-hydro, all of these different quantities are easily computed via

.. code-block:: python

    network = ekh.river_network.load("efas", "5")
    locations = {
        "station1": (10, 10),
        "station2": (10, 10),
        "station3": (10, 10)
    }

    # lengths take node-level information
    field = np.random.rand(network.n_nodes)
    max_length = ekh.length.max(network, locations, field)
    min_length = ekh.length.min(network, locations, field)

    # distances take edge-level information
    field = np.random.rand(network.n_edges)
    max_distance = ekh.distance.max(network, locations, field)
    min_distance = ekh.distance.min(network, locations, field)

Directed and undirected distances or lengths
--------------------------------------------

By default, distances and lengths are calculated downstream only. However, some use cases may be interested in upstream distances/lengths, or undirected lengths/distances.
This is easily specified by the `upstream` and `downstream` arguments.

.. code-block:: python

    min_length_upstream = ekh.length.min(network, locations, field, upstream=True, downstream=False)
    min_length_downstream = ekh.length.min(network, locations, field, upstream=False, downstream=True)
    min_length_undirected = ekh.length.min(network, locations, field, upstream=True, downstream=True)

As shorthands, earthkit-hydro also provides the means of automatically computing starting from the sources or the sinks with

.. code-block:: python

    ekh.length.to_sink(network, field, path="shortest")
    ekh.length.to_source(network, field, path="shortest")
    ekh.distance.to_sink(network, field, path="shortest")
    ekh.distance.to_source(network, field, path="shortest")

Longest path versions are also available with `path="longest"`.
