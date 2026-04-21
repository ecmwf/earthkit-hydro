Coordinate systems
==================

Spatial data requires careful attention to coordinate reference systems (CRS).
This page explains how earthkit-hydro handles coordinates and what you need to know.

Importance of coordinate systems
---------------------------------

River networks and input data must use compatible coordinate systems for operations like:

- Locating monitoring stations
- Extracting catchment boundaries
- Overlaying with other spatial data
- Calculating distances

Mismatched coordinate systems lead to incorrect results or errors.

Supported systems
-----------------

earthkit-hydro works with:

- **Geographic coordinates** (latitude/longitude)
- **Projected coordinates** (meter-based grids like UTM)
- **Custom grids** (model-specific coordinate systems)

The library relies on coordinate information from your data source (NetCDF attributes, GeoTIFF metadata, etc.).

Best practices
--------------

**Check your CRS:** Always verify that input locations use the same CRS as your river network

**Use metadata:** Store CRS information in your data files (NetCDF CF conventions, GeoTIFF tags)

**Reproject when needed:** Use tools like ``rasterio`` or ``pyproj`` to reproject coordinate data before using with earthkit-hydro

**Document assumptions:** Note which CRS you're using in analysis scripts

See also
--------

- :doc:`../howto/specify_locations` - Practical guide to specifying locations correctly
- :doc:`river_network_concepts` - Understanding river network representation
