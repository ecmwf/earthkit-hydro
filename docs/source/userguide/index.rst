User Guide
==========

**earthkit-hydro** is designed to simplify the process of working with hydrological data, providing a range of tools for catchment delineation, river network analysis, and more. It supports various data formats and array backends, making it versatile for different applications.

At its core, **earthkit-hydro** is a library for conducting operations on river networks. A typical workflow involves:

1. Loading a river network
2. Performing operations on the network, such field propagation, catchment averages, distance calculations and more.
3. Saving or plotting the results

In this user guide, we provide detailed instructions for such steps.

Basics:

.. toctree::
   :maxdepth: 200
   :titlesonly:

   loading_a_river_network
   xarray_array_backend
   raster_vector_inputs

Operations:

.. toctree::
   :maxdepth: 200
   :titlesonly:

   flow_accumulations
   specifying_locations
   catchment_delineation
   catchment_statistics
   distance_length_calculations
   subnetwork_creation

Misc:

.. toctree::
   :maxdepth: 200
   :titlesonly:

   earthkit
   pcraster
