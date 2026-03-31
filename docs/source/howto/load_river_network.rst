Loading a river network
=======================

earthkit-hydro supports multiple river network formats, including PCRaster, CaMa-Flood, HydroSHEDS, MERIT-Hydro and GRIT.

Pre-computed river networks
---------------------------

Many commonly used river networks are available pre-computed. This is the recommended way to load a network:

.. code-block:: python

    import earthkit.hydro as ekh

    # Load the EFAS version 5 river network
    network = ekh.river_network.load("efas", "5")

For a full list of available networks, see the API reference :doc:`../autodocs/earthkit.hydro.river_network`.

Custom river networks
---------------------

If a river network is not available via ``load``, you can create one from a file in any supported format:

.. code-block:: python

    network = ekh.river_network.create(path, river_network_format, source)

This operation involves topologically sorting the network, which is computationally expensive for large grids. It is therefore recommended to export the result for re-use:

.. code-block:: python

    network.export("my_river_network.joblib")

In subsequent analyses, the pre-computed network can be loaded directly:

.. code-block:: python

    network = ekh.river_network.create("my_river_network.joblib", "precomputed")

See also
--------

- :doc:`../tutorials/loading_river_networks` — Tutorial walkthrough
- :doc:`../explanation/river_network_concepts` — How river networks are represented
- :doc:`../autodocs/earthkit.hydro.river_network` — API reference
