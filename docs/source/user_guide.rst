User Guide
==========

This section walks through a typical workflow using :mod:`earthkit.hydro`. It is
based on the ``docs/notebooks/example.ipynb`` notebook.

Fields and River Networks
-------------------------

A river network is defined on a grid. All fields passed to ``earthkit-hydro``
should be on the same grid. Multidimensional fields are also supported, with the
last two axes representing the grid.

River Network Creation
----------------------

Precomputed networks can be listed using ``river_network.available()`` and loaded
with ``river_network.load``:

.. code-block:: python

   import earthkit.hydro as ekh

   # list available networks
   ekh.river_network.available()

   # load the EFAS v5 river network
   rn = ekh.river_network.load(domain="efas", river_network_version="5")

Upstream Metrics
----------------

Given a river network, upstream metrics can be computed via the
:mod:`earthkit.hydro.upstream` module:

.. code-block:: python

   result = ekh.upstream.sum(rn, field)

Other functions such as ``mean``, ``max`` and ``min`` are available. Lower level
access is provided by ``calculate_upstream_metric``.

Finding Catchments and Subcatchments
------------------------------------

Catchments or subcatchments defined by points can be found with
``catchments.find`` or ``subcatchments.find``:

.. code-block:: python

   catchments = ekh.catchments.find(rn, points)
   subcatchments = ekh.subcatchments.find(rn, points)

Computing Subcatchment Metrics
------------------------------

Metrics over the subcatchments of given points are computed using the
``subcatchments`` module:

.. code-block:: python

   values = ekh.subcatchments.mean(rn, field, points)

Zonal Metrics
-------------

For arbitrary areas not defined by a catchment, ``zonal.*`` functions compute a
metric over labelled zones:

.. code-block:: python

   zone_values = ekh.zonal.mean(field, labels)

For more details and visual examples, see ``docs/notebooks/example.ipynb``.

Distances and Lengths
---------------------

Distances or lengths along a river network can be computed with the
:mod:`earthkit.hydro.distance` and :mod:`earthkit.hydro.length` modules. The
``min`` and ``max`` functions take a river network, a set of points and optional
weights representing the distance (or length) to the downstream cell.

.. code-block:: python

   # minimum upstream distance from ``points``
   dist = ekh.distance.min(rn, points, upstream=True, downstream=False)

   # maximum downstream length using predefined cell lengths
   length = ekh.length.max(rn, points, weights)

Convenience helpers ``distance.to_source`` and ``distance.to_sink`` (and their
``length`` counterparts) compute shortest or longest paths to all sources or
sinks.

Moving Fields Upstream and Downstream
------------------------------------

Fields can be propagated along a river network using ``move_downstream`` or
``move_upstream``. This updates each node with the value from the connected
upstream or downstream node.

.. code-block:: python

   downstream_field = ekh.move_downstream(rn, field)
   upstream_field = ekh.move_upstream(rn, field)

Flow Accumulation
-----------------

For more advanced operations one can apply an arbitrary ``ufunc`` along the
network using ``flow_downstream`` or ``flow_upstream``. This is a generalisation
of the upstream and downstream metric routines.

.. code-block:: python

   accumulated = ekh.flow_downstream(rn, field, ufunc=np.add)

These functions underlie the higher level metric helpers discussed above.
