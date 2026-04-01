Flow direction systems
======================

Different river network formats use different conventions for encoding flow directions.
This page explains these systems and how earthkit-hydro handles them.

Overview
--------

Flow direction encoding determines how a grid cell's outflow direction is represented numerically.
Understanding these systems helps when working with different river network datasets.

D8 method
---------

The D8 (8-direction) method allows flow from a cell to one of its 8 neighbors:

- North (N)
- Northeast (NE)
- East (E)
- Southeast (SE)
- South (S)
- Southwest (SW)
- West (W)
- Northwest (NW)

Different formats encode these directions differently:

**PCRaster encoding:**

.. code-block:: text

    7  8  9
    4  5  6
    1  2  3

Value 5 indicates a pit (no outflow).

**ArcGIS encoding (powers of 2):**

.. code-block:: text

    32  64  128
    16   0    1
     8   4    2

Value 0 indicates a pit.

**Sequential indices:** Some formats use 0-7 indexing, often following:

.. code-block:: text

    5  6  7
    4  X  0
    3  2  1

Automatic conversion
--------------------

When you load a river network, earthkit-hydro automatically converts the native encoding to its internal representation. You typically don't need to worry about these differences unless you're:

- Creating custom river networks
- Debugging flow direction issues
- Interfacing with external tools

Special cases
-------------

**Pits and sinks:** Cells with no outflow (ocean, lakes, or depression sinks)

**Edge cells:** Cells at domain boundaries may need special handling

**Bifurcations:** Some formats support flow splitting to multiple cells

See also
--------

- :doc:`river_network_concepts` - General river network concepts
- :doc:`../howto/load_river_network` - Loading different formats
