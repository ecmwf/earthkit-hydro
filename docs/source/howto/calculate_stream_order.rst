Streamorder
===========

Calculating the streamorder of a river network can provide insights into the hierarchy, importance and structure of the river system.

In earthkit-hydro, two common streamorder methods are implemented: Strahler and Shreve. These are calculated using the following functions:

.. code-block:: python

    network = ekh.river_network.load("efas", "5")

    strahler_order = ekh.streamorder.strahler(network, return_type="gridded")
    shreve_order = ekh.streamorder.shreve(network, return_type="gridded")

Note that these are topological properties of the river network and do not depend on any external field.
