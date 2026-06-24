Frequently Asked Questions
==========================


General
-------

**What river network formats does earthkit-hydro support?**

PCRaster D8, ArcGIS/ESRI D8, CaMa-Flood, HydroSHEDS, MERIT-Hydro and GRIT formats are all supported. See :doc:`autodocs/earthkit.hydro.river_network` for the full list of formats and pre-computed networks.

**What array backends are supported?**

NumPy, xarray, CuPy, PyTorch, JAX, MLX and TensorFlow. The default backend is NumPy. You can switch backends via ``network.to_device(array_backend=chosen_array_backend)``. See :doc:`howto/use_different_array_backends` for details.

**Does earthkit-hydro support GPU acceleration?**

Yes. Any backend with GPU support (CuPy, PyTorch, JAX, etc.) can be used directly. Load or convert a river network to the desired backend and all subsequent operations run on the GPU.

**Can I use earthkit-hydro with xarray datasets?**

Yes. All top-level functions accept xarray and return xarray objects.

**Does earthkit-hydro handle bifurcating river networks?**

Yes.

**Why am I getting "Index is out of bounds"?**

Data passed to functions is assumed to be already on the same grid as the river network you are using. Check that your data is indeed on the same grid. If you need to regrid, please refer to earthkit-geo.

Installation
------------

**What Python version do I need?**

We adopt stable Python versions (see https://devguide.python.org/versions/).

**How do I install GPU support?**

Install the GPU backend of your choice separately (e.g. ``pip install torch``), then convert your river network with ``network.to_device(array_backend=chosen_array_backend, device=chosen_device)``.


Data and performance
--------------------

**Loading a custom river network is slow. How can I speed it up?**

Creating a river network from a raw flow direction file requires topological sorting, which is expensive for large grids. Export the processed network once with ``network.export("my_network.joblib")`` and reload it with ``ekh.river_network.create("my_network.joblib", "precomputed")``. See :doc:`howto/load_river_network` for details.

**How are missing values handled?**

earthkit-hydro follows the NumPy convention: missing values are represented as ``np.nan`` and propagate through all operations. See :doc:`concepts/missing_value_handling` for the rationale behind this design.

**I'm migrating from PCRaster. Where do I start?**

See the :doc:`concepts/pcraster_compatibility` page for a function-by-function translation table and a summary of the key differences.

**What is the difference between distance and length?**

In earthkit-hydro, *distances* are edge properties (the cost of traversing a connection between two nodes) while *lengths* are node properties (the extent associated with each node). This distinction matters at confluences and bifurcations. See :doc:`concepts/distance_vs_length_concepts` for a full explanation.
