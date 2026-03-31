Optimising performance
======================

Cache river networks
--------------------

Creating a river network from a raw flow direction file requires topological sorting, which is the most expensive step. Export the result and reload it in subsequent runs:

.. code-block:: python

    import earthkit.hydro as ekh

    # First time: create and export
    network = ekh.river_network.create("my_network.nc", "pcr_d8", "file")
    network.export("my_network.joblib")

    # Subsequent runs: 100–1000x faster
    network = ekh.river_network.create("my_network.joblib", "precomputed", "file")

Pre-computed networks loaded via ``ekh.river_network.load`` are already optimised.

GPU acceleration
----------------

For large domains (> 1M cells), moving to a GPU backend can give significant speedups:

.. code-block:: python

    import cupy as cp

    network = ekh.river_network.load("efas", "5").to_device(array_backend="cupy")
    field_gpu = cp.asarray(field)
    result_gpu = ekh.upstream.sum(network, field_gpu)

    # Move result back to CPU if needed
    result = cp.asnumpy(result_gpu)

PyTorch, JAX, and other GPU-capable backends work the same way.

Reduce network size for testing
-------------------------------

Extract a regional subnetwork for faster development cycles:

.. code-block:: python

    mask = (lats > 40) & (lats < 50) & (lons > 0) & (lons < 10)
    small_network = ekh.subnetwork.from_mask(network, mask)

Use float32
-----------

Most analyses do not need float64 precision. Halving the data type halves memory usage:

.. code-block:: python

    import numpy as np

    field = np.ones(network.shape, dtype=np.float32)
    result = ekh.upstream.sum(network, field)  # also float32

See also
--------

- :doc:`../explanation/performance_considerations` — Performance characteristics in depth
- :doc:`../explanation/array_backend_design` — Choosing the right backend
- :doc:`use_different_array_backends` — Switching backends
