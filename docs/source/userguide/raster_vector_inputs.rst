Raster and vector networks
==========================

earthkit-hydro supports both raster (gridded) and vector river networks. These network types have slightly different characteristics which necessitate some clarification.

.. raw:: html

   <div style="float: center;">

.. figure:: ../../images/raster_vector_networks.jpg
   :width: 400px

   *From:* :cite:`rastervector`

.. raw:: html

   </div>

Raster networks
---------------

Raster networks are the most common type of river network in earthkit-hydro. They are represented as a grid where each cell corresponds to a specific location in the river network.
With these type of networks, it is often most natural to conduct river network operations directly on this grid.

.. code-block:: python

    import earthkit.hydro as ekh

    # Load a raster river network
    network = ekh.river_network.load("efas", "5")

    field = np.ones(network.shape) # field on the river network grid

    output = ekh.upstream.sum(network, field) # output on same grid

Vector networks
---------------

In vector networks, each river segment is represented as a node, and the network is defined by the connections between these nodes.

.. code-block:: python

    # vector field (1D) defined on the nodes of the river network
    field = np.ones(network.n_nodes)

    # output field is also 1D, defined on the nodes of the river network
    output = ekh.upstream.sum(network, field, return_grid=False)

Switching between vector and raster
-----------------------------------

Raster networks can also be used as if they were vector networks, since internally the raster network is represented as a vector network.
To allow users to work with both types of networks seamlessly, earthkit-hydro has the `return_grid` function argument, which is by default True.
This allows the user to specify whether they want a gridded output (only available for raster networks), or a 1d-vector output.

Note that it is possible to mix gridded and raster inputs to functions.

.. code-block:: python

    # vector field (1D) defined on the nodes of the river network
    field = np.ones(network.n_nodes)

    # output field is a grid
    output = ekh.upstream.sum(network, field)

Multidimensional inputs
-----------------------

Any leading dimensions of the data are treated as batch/vectorised dimensions, allowing for operations on multiple fields at once.
This means that users can pass directly time series or other multi-dimensional data without needing to manually loop.
The last dimensions are always assumed to be spatial i.e. either the grid dimensions, or the 1d vector dimension.

.. code-block:: python

    # vector field (1D) defined on the nodes of the river network
    field = np.ones((3, 4, 5, network.n_nodes))

    # output field is of shape (3, 4, 5, *network.shape)
    output = ekh.upstream.sum(network, field)
