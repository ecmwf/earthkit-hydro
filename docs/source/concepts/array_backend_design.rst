Array backend design
====================

This page explains why earthkit-hydro supports multiple array backends and how this design choice affects your work.

The flexibility challenge
-------------------------

Hydrological datasets come in many forms - NetCDF files, GeoTIFFs, CSV tables, databases - and scientists use different computing ecosystems:

- Climate scientists often use **xarray** for labeled multi-dimensional arrays
- Machine learning practitioners use **PyTorch**, **JAX**, or **TensorFlow**
- Performance-focused users leverage **CuPy** for GPU acceleration
- Traditional scientific computing relies on **NumPy**

Rather than forcing everyone into one framework, earthkit-hydro is **backend-agnostic**.

How it works
------------

Earthkit in general, but specifically also earthkit-hydro operations work with any array backend that supports basic operations like indexing, aggregation, and mathematical operations. The library:

1. **Detects** what array type you provide (numpy, cupy, torch, etc.)
2. **Dispatches** operations to backend-appropriate implementations
3. **Returns** results in the same array type you provided

This means:

.. code-block:: python

    import numpy as np
    import torch
    import earthkit.hydro as ekh

    network = ekh.river_network.load("efas", "5")

    # Works with NumPy
    numpy_data = np.ones(network.shape)
    result_np = ekh.upstream.array.sum(network, numpy_data)  # Returns NumPy array

    # Works with PyTorch
    torch_data = torch.ones(network.shape)
    result_torch = ekh.upstream.array.sum(network, torch_data)  # Returns PyTorch tensor

No conversion needed, no backend lock-in.

Benefits for machine learning
------------------------------

Supporting ML frameworks natively enables:

**Differentiability:** Operations with PyTorch/JAX are differentiable, allowing:

- Gradient-based optimization of hydrological models
- Integration of physical constraints in neural networks
- Parameter estimation through backpropagation

**GPU acceleration:** Automatic GPU execution with CuPy/PyTorch CUDA tensors for:

- Processing large spatial domains
- Ensemble simulations
- Real-time applications

**Framework integration:** Seamless use in existing ML pipelines without data conversion overhead.

Benefits for traditional workflows
-----------------------------------

Even if you don't use ML frameworks, backend flexibility provides:

**xarray integration:** Preserve dimension labels, coordinates, and metadata throughout your workflow:

.. code-block:: python

    import earthkit.data as ekd
    import earthkit.hydro as ekh

    # Input as xarray with coordinates and metadata
    runoff = ekd.from_source("file", "runoff.nc").to_xarray()["runoff"]

    network = ekh.river_network.load("efas", "5")

    discharge = ekh.upstream.sum(network, runoff)
    # discharge is still an xarray DataArray with coordinates!

**Performance portability:** Switch to GPU execution by changing array type, not code:

.. code-block:: python

    # CPU version
    import numpy as np
    data = np.array(...)
    result = ekh.upstream.array.sum(network, data)

    # GPU version - same operation, just different array type
    import cupy as cp
    data = cp.array(...)
    result = ekh.upstream.array.sum(network, data)  # Runs on GPU!

See also
--------

- :doc:`../howto/use_different_array_backends` - Practical guide to using different backends
- :doc:`../tutorials/array_backend` - Tutorial on array backend usage
- :doc:`performance_considerations` - When to choose which backend for performance
