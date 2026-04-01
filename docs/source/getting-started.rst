Installation and Getting Started
================================

Installation
------------

Install earthkit-hydro from PyPI:

.. code-block:: bash

   pip install earthkit-hydro==1.0.0rc0

For GPU support, also install the desired array backend e.g. cupy, torch etc..


Quick start
-----------

earthkit-hydro works by loading a river network and then performing hydrological operations on it.

**1. Load a river network**

.. code-block:: python

    import earthkit.hydro as ekh

    network = ekh.river_network.load("efas", "5")

This loads the EFAS (European Flood Awareness System) version 5 river network. Several pre-computed networks are available (EFAS, GloFAS, CaMa-Flood, HydroSHEDS, MERIT-Hydro, GRIT).

**2. Compute a flow accumulation**

.. code-block:: python

    import numpy as np

    # A field of ones: the upstream sum gives the number of upstream cells
    field = np.ones(network.n_nodes)
    upstream_area = ekh.upstream.sum(network, field)

**3. Find catchments**

.. code-block:: python

    # Specify outlet locations by coordinate
    locations = {"outlet_A": (48.0, 12.0), "outlet_B": (50.0, 8.0)}

    catchments = ekh.catchments.find(network, locations)

**4. Compute catchment statistics**

.. code-block:: python

    catchment_means = ekh.catchments.mean(network, field, locations)


What next?
----------

- :doc:`tutorials/index` — Work through hands-on notebooks covering each feature.
- :doc:`howto/index` — Find recipes for specific tasks.
- :doc:`explanation/index` — Understand the core concepts and design decisions.
- :doc:`autodocs/earthkit.hydro` — Full API reference.
