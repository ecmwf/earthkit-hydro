Loading a river network
=======================

earthkit-hydro provides a straightforward way to load river networks from various formats. The library supports multiple river network formats, including those used by PCRaster, CaMa-Flood, HydroSHEDS, and MERIT-Hydro.

Many river networks are commonly used for hydrological analysis and modelling, such as the EFAS river network. earthkit-hydro provides precomputed versions of such river networks which are available via

.. code-block:: python

    import earthkit.hydro as ekh

    # Load the EFAS version 5 river network
    network = ekh.river_network.load("efas", "5")

This is the most convenient and performant way to load a river network, and is therefore recommended for most users. For a full list of networks, view the API reference :doc:`../autodocs/earthkit.hydro.river_network`.

Custom river networks
---------------------
If a river network is not available via `ekh.river_network.load`, it is possible to create a custom river network from scratch. Many different formats and sources are supported, as detailed in the API reference :doc:`../autodocs/earthkit.hydro.river_network`.

.. code-block:: python

    network = ekh.river_network.create(path, river_network_format, source)

This operation involves topologically sorting the river network, which is computationally expensive for large networks. Therefore, it is recommended to export the river network for re-use.

.. code-block:: python

    network.export("my_river_network.joblib")

In subsequent analyses, the precomputed river network can now be loaded via

.. code-block:: python

    network = ekh.river_network.create("my_river_network.joblib", "precomputed")
