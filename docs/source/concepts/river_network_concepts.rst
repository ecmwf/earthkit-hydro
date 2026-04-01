River network concepts
======================

This page explains how river networks are represented in earthkit-hydro and why this representation matters.

What is a river network?
-------------------------

In earthkit-hydro, a river network is a digital representation of how water flows across a landscape. Rather than storing every detail of every stream, river networks use a structured grid where:

- Each grid cell (or node) represents a location
- Flow directions indicate which cells drain into which neighbors
- Topological relationships capture the network structure

This representation enables efficient computation of hydrological properties across entire basins.

Flow direction encoding
-----------------------

River networks store flow direction using different encoding schemes depending on the data source:

**D8 (8-direction):** Each cell flows to one of its 8 neighbors (N, NE, E, SE, S, SW, W, NW). This is common in DEM-based routing schemes.

**Sequential encoding:** Flow direction stored as integer codes. Different formats use different encoding schemes:

- PCRaster: 1-9 with 5 representing pits
- ArcGIS: Powers of 2 (1, 2, 4, 8, 16, 32, 64, 128)
- CaMa-Flood: Custom sequential indices

earthkit-hydro automatically handles these different encodings when you specify the appropriate format.

Topological sorting
-------------------

When you load or create a river network, earthkit-hydro performs **topological sorting** - ordering cells from upstream to downstream. This one-time computation enables:

- Fast accumulation operations
- Efficient catchment delineation
- Vectorized propagation algorithms

This is why creating custom networks can be slow (requires sorting), but once created, operations are fast.

Supported formats
-----------------

earthkit-hydro supports multiple river network formats:

**PCRaster:** Common in European hydrological modeling (e.g., LISFLOOD, EFAS)

**CaMa-Flood:** Global-scale river routing with explicit channel representation

**HydroSHEDS:** Global hydrographic datasets at multiple scales

**MERIT-Hydro:** High-resolution global hydrography

**GRIT:** Graph-based river topology

Each format has specific conventions for encoding flow direction, handling edge cells, and representing special features like pits or multi-flow directions.

Bifurcations
------------

Most river networks represent convergent flow (many tributaries → one downstream), but some systems include **bifurcations** where flow splits:

- River deltas
- Distributary systems
- Artificial diversions

earthkit-hydro includes specialized support for bifurcations in compatible formats (e.g., CaMa-Flood), enabling more realistic representation of these features.

Resolution and scale
--------------------

River networks are available at different resolutions:

- **High resolution** (< 100m): Detailed local analysis, large data volumes
- **Medium resolution** (1-5 km): Regional modeling, balanced computation
- **Coarse resolution** (> 10 km): Continental/global scale, fast computation

Resolution choice affects:

- Catchment boundary precision
- Small stream representation
- Computational requirements
- Available datasets

Why pre-computed networks?
---------------------------

earthkit-hydro provides pre-computed versions of common river networks (like EFAS) because:

1. **Performance:** Topological sorting is already done
2. **Consistency:** Everyone uses the same reference network
3. **Convenience:** No need to source and process raw files
4. **Validation:** Pre-computed networks are tested and verified

For custom networks, you can export your own pre-computed version after initial creation to get the same benefits.

See also
--------

- :doc:`../howto/load_river_network` - How to load different river network formats
- :doc:`flow_direction_systems` - Detailed comparison of flow direction encodings
- :doc:`coordinate_systems` - Working with different spatial reference systems
