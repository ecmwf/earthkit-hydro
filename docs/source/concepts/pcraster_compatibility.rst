PCRaster compatibility
======================

If you're familiar with PCRaster, this guide explains how earthkit-hydro relates to it and helps you translate concepts.

Why earthkit-hydro differs from PCRaster
-----------------------------------------

PCRaster has been a cornerstone of hydrological modeling for decades. earthkit-hydro takes a different approach:

**Design philosophy:**

- **PCRaster:** Comprehensive GIS-like environment with its own data format
- **earthkit-hydro:** Focused library that integrates with the scientific Python ecosystem

**Performance characteristics:**

- **PCRaster:** C++ backend, mature optimizations
- **earthkit-hydro:** Vectorized Python operations, often faster for large datasets

**Ecosystem integration:**

- **PCRaster:** Standalone tool with Python bindings
- **earthkit-hydro:** Native Python, works with NumPy/xarray/PyTorch/etc.

When to use each
----------------

**Use PCRaster if:**

- You have existing PCRaster workflows
- You need PCRaster-specific functionality
- You work primarily with PCRaster map formats

**Use earthkit-hydro if:**

- You want better integration with Python scientific stack
- You need GPU acceleration or ML framework integration
- You want faster performance for large-scale operations
- You work with xarray or other modern data structures

**Use both:**

earthkit-hydro can work with PCRaster-format river networks, so you can combine tools as needed.

Function mapping
----------------

Here's how common PCRaster operations translate to earthkit-hydro:

+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| **PCRaster**     | **earthkit-hydro**     | **Note**                                                                                                                |
+==================+========================+=========================================================================================================================+
| accuflux         | upstream.sum           |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| catchmenttotal   | upstream.sum           |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| downstream       | move.upstream          |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| upstream         | move.downstream        |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| catchment        | catchments.find        |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| subcatchment     | catchments.find        | overwrite=False                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| path             | upstream.max           |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| ldddist          | distance.min           | friction input is slightly different from weights; by default, distance between two nodes is one regardless of diagonal |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| downstreamdist   | distance.to_sink       | Same caveats as for ldddist                                                                                             |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| slopelength      | distance.to_source     | path="longest"; same caveats as for ldddist                                                                             |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| lddmask          | subnetwork.from_mask   |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+
| abs, sin, cos,   | np.abs, np.sin,        | Any array operations can be directly used (example shown for NumPy backend)                                             |
| tan, ...         | np.cos, np.tan, ...    |                                                                                                                         |
+------------------+------------------------+-------------------------------------------------------------------------------------------------------------------------+


Key differences to keep in mind
--------------------------------

**Missing value handling:**

earthkit-hydro treats missing values as ``np.nan``. Any arithmetic involving a missing value returns a missing value. PCRaster's missing value behavior can differ in some operations.

**Vector field support:**

earthkit-hydro can work with vector (multi-dimensional) fields, while PCRaster operations are typically scalar.

**API philosophy:**

earthkit-hydro uses a functional API (pass data to functions) rather than PCRaster's map algebra syntax. This means slightly different code structure even for equivalent operations.

See also
--------

- :doc:`../howto/load_river_network` - Loading PCRaster-format networks
- :doc:`river_network_concepts` - Understanding river network representation
- :doc:`flow_direction_systems` - Flow direction encoding details
