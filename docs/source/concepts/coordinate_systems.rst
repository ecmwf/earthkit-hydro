Coordinate systems
==================

earthkit-hydro assumes that all input data are defined on the same grid as the river network. This means that the coordinates and grid structure of forcing data, masks, and locations must be consistent with the river network being used.

Why this matters
----------------

Many operations in earthkit-hydro depend on a direct correspondence between the river network and the input data, including:

* Locating monitoring stations
* Extracting catchment boundaries
* Associating data values with river reaches
* Calculating upstream and downstream relationships

If the input data are on a different grid, these operations may produce incorrect results or fail entirely.

Grid compatibility
------------------

earthkit-hydro does not automatically reproject or regrid data. Instead, it assumes that:

* Input datasets use the same grid as the river network
* Grid coordinates correspond to the river network coordinates
* Any required coordinate transformations have already been applied

The underlying coordinate reference system (CRS) may be geographic, projected, or model-specific, provided that both the river network and the input data use the same grid definition.

To regrid data, we recommend using **earthkit-geo**.

Best practices
--------------

**Verify grid consistency:** Check that your input data and river network share the same grid and coordinate definitions.

**Preserve metadata:** Store coordinate and CRS information in your data files whenever possible.

**Regrid before use:** If your data are on a different grid, regrid them to the river network grid using appropriate tools such as earthkit-geo before using them with earthkit-hydro.

**Document assumptions:** Record any regridding or coordinate transformations applied during preprocessing.

See also
--------

- :doc:`../howto/specify_locations` - Practical guide to specifying locations correctly
- :doc:`river_network_concepts` - Understanding river network representation
