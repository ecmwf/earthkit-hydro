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

earthkit-hydro operations work with any array backend that supports basic operations like indexing, aggregation, and mathematical operations. The library:

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
    result_np = ekh.accumulate(network, numpy_data)  # Returns NumPy array

    # Works with PyTorch
    torch_data = torch.ones(network.shape)
    result_torch = ekh.accumulate(network, torch_data)  # Returns PyTorch tensor

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

    import xarray as xr
    import earthkit.hydro as ekh

    # Input as xarray with coordinates and metadata
    runoff = xr.open_dataset("runoff.nc")["runoff"]

    network = ekh.river_network.load("efas", "5")

    # Output preserves xarray structure
    discharge = ekh.accumulate(network, runoff)
    # discharge is still an xarray DataArray with coordinates!

**Performance portability:** Switch to GPU execution by changing array type, not code:

.. code-block:: python

    # CPU version
    import numpy as np
    data = np.array(...)
    result = ekh.accumulate(network, data)

    # GPU version - same operation, just different array type
    import cupy as cp
    data = cp.array(...)
    result = ekh.accumulate(network, data)  # Runs on GPU!

Trade-offs and limitations
--------------------------

**Not all operations possible on all backends:** Some backends lack certain features. For example:

- JAX arrays are immutable, limiting in-place operations
- Some sparse operations aren't available in all frameworks

earthkit-hydro will raise clear errors when operations aren't supported.

**Performance varies:** While all backends work, performance characteristics differ:

- NumPy: Mature, CPU-optimized
- CuPy: Drop-in GPU acceleration
- PyTorch: Optimized for ML, good GPU support
- JAX: JIT compilation can be very fast

**Memory layout matters:** Each framework has preferred memory layouts. earthkit-hydro tries to respect these, but cross-framework conversion can be expensive.

Choosing a backend
------------------

**Use NumPy if:** You want simplicity, wide compatibility, and CPU-based computation

**Use xarray if:** You work with labeled multi-dimensional earth science data

**Use CuPy if:** You need GPU acceleration without PyTorch/TensorFlow overhead

**Use PyTorch if:** You're integrating with ML models or need autograd

**Use JAX if:** You want JIT compilation and functional programming benefits

**Use TensorFlow if:** You're already in TensorFlow ecosystem

Implementation details
----------------------

earthkit-hydro uses **array protocols** rather than explicit type checking:

- Duck typing for compatibility
- ``__array_namespace__`` support for Array API standard
- Minimal backend-specific code

This approach future-proofs the library as new array libraries emerge.

See also
--------

- :doc:`../howto/use_different_array_backends` - Practical guide to using different backends
- :doc:`../tutorials/array_backend` - Tutorial on array backend usage
- :doc:`performance_considerations` - When to choose which backend for performance
