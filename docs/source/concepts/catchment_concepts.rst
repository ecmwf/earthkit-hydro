Catchment concepts
==================

This page explains the fundamental concepts behind catchment delineation in earthkit-hydro.

What is a catchment?
--------------------

A catchment (also called a watershed or drainage basin) is the area of land where all water flows to a single outlet point. Understanding catchments is fundamental to hydrological analysis because they define the spatial extent of water movement in a landscape.

When rain falls within a catchment, it eventually flows to the catchment outlet, either as surface runoff or through subsurface pathways. This makes catchments natural units for studying water balance, flood routing, and water resource management.

How catchment delineation works
--------------------------------

In earthkit-hydro, catchment delineation works by tracing flow paths upstream from specified outlet locations. The process:

1. **Specify outlet locations** - Define one or more points where you want to delineate catchments
2. **Trace upstream** - Follow flow directions backwards to identify all cells that drain to each outlet
3. **Label cells** - Assign each cell a label corresponding to its outlet

.. image:: ../../images/catchment.gif
   :width: 400px
   :align: center

*Animation showing catchment delineation from multiple outlets. Cells are progressively labeled as the algorithm traces upstream from outlet points.*

Overlapping catchments
-----------------------

When multiple outlet points are specified, their catchments may overlap (i.e., one outlet is upstream of another). By default, earthkit-hydro resolves this by giving priority to the **most downstream outlet**.

This behavior means:

- Upstream catchments are absorbed into downstream ones
- Only non-overlapping catchment portions remain labeled for upstream points
- The result shows distinct catchment boundaries

This default is useful when you want to identify major basin boundaries without double-counting areas.

Subcatchments
-------------

Sometimes you want to preserve boundaries for all specified outlets, even when they overlap. Setting ``overwrite=False`` enables subcatchment delineation:

.. image:: ../../images/subcatchment.gif
   :width: 400px
   :align: center

*Animation showing subcatchment delineation where each outlet retains its full upstream area, even when catchments overlap.*

This mode is useful for:

- Identifying contributing areas to specific monitoring stations
- Analyzing nested catchments at different scales
- Understanding how upstream changes affect downstream locations

Practical considerations
------------------------

**Choice of outlets:** Where you place outlet points significantly affects results. Common choices include:

- River gauging stations for calibration
- River mouths for full basin analysis
- Confluences for tributary analysis
- Points of interest like water intakes or discharge locations

**Coordinate systems:** Ensure your outlet locations use the same coordinate reference system as your river network to avoid misalignment.

**Resolution effects:** Catchment boundaries depend on the resolution of your river network. Coarser networks produce more generalized boundaries.

See also
--------

- :doc:`../howto/delineate_catchments` - Practical guide to delineating catchments
- :doc:`../howto/specify_locations` - How to specify outlet locations correctly
- :doc:`river_network_concepts` - Understanding river network representation
