Catchment delineation
=====================

A common task in hydrology is identifying the catchment area for a given point in a river network.
This process, known as catchment delineation, involves determining the area that drains into a specific point.

.. image:: ../../images/catchment.gif
   :width: 250px
   :align: right

In earthkit-hydro, this is accomplished by specifying certain start locations, and labelling all nodes flowing towards those start locations.
If start locations belong to the same catchment, the node furthest downstream takes priority and overwrites any upstream start locations.

This can be done in earthkit-hydro using the `catchments.find` method.

.. raw:: html

   <br style="clear: both">

.. code-block:: python

    network = ekh.river_network.load("efas", "5")

    labelled_field = ekh.catchments.find(network, locations)

Subcatchments can also be found by making use of the `overwrite` keyword.

.. image:: ../../images/subcatchment.gif
   :width: 250px
   :align: left

.. code-block:: python

    labelled_field = ekh.catchments.find(
                                       network,
                                       locations,
                                       overwrite=False
                                       )
