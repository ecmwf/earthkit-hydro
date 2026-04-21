Performance considerations
==========================

Understanding performance characteristics helps you make informed decisions about how to use earthkit-hydro efficiently.

One-time costs vs. repeated operations
---------------------------------------

**Creating/loading networks:**

- Loading pre-computed networks: Fast (seconds)
- Creating custom networks: Slow (minutes to hours for large networks)
- Reason: Topological sorting required for custom networks

**Recommendation:** Export and reuse custom networks

**Running operations:**

Once a network is loaded, operations are highly optimized:

- Accumulations: Vectorized, single pass through sorted network
- Catchment delineation: Graph traversal, scales with network size
- Statistics: Depends on aggregation method and data size

Resolution and domain size
---------------------------

Performance scales roughly with:

- Number of cells in domain
- Number of time steps (for temporal data)
- Complexity of operation

**Example scaling:**

- 1 million cells: Sub-second operations
- 10 million cells: Seconds
- 100+ million cells: May benefit from GPU or chunking

Array backend performance
--------------------------

Different backends have different performance characteristics:

**NumPy (CPU):**
- Mature, well-optimized
- Single-threaded for most operations
- Good for moderate problem sizes

**CuPy (GPU):**
- Major speedup for large datasets
- Requires GPU with appropriate VRAM
- Best for repeated operations on large grids

**PyTorch (CPU or GPU):**
- Similar to NumPy on CPU
- Good GPU performance
- Overhead from autograd if not using ``torch.no_grad()``

**JAX:**
- JIT compilation can provide speedups
- Good for repeated operations with same shapes
- Initial JIT compilation has overhead

Memory considerations
----------------------

**Rule of thumb:** Expect memory usage of 2-3x your input data size during operations

**Strategies for large datasets:**

- Process time steps sequentially rather than all at once
- Use chunking with xarray/dask
- Choose memory-efficient data types (e.g., float32 vs. float64)
- Stream data from disk rather than loading all at once

**GPU memory:** More limited than system RAM - monitor VRAM usage with ``nvidia-smi``

Optimization strategies
-----------------------

**For repeated analyses:**

1. Pre-compute and cache networks
2. Reuse allocated arrays when possible
3. Batch operations when feasible

**For large domains:**

1. Consider spatial chunking
2. Use GPU backends (CuPy, PyTorch with CUDA)
3. Process temporal data in chunks

**For ML workflows:**

1. Use ``torch.no_grad()`` when gradients not needed
2. Batch multiple scenarios together
3. Consider mixed precision (float16/float32)

Profiling your code
-------------------

To identify bottlenecks:

**Python profiling:**

.. code-block:: python

    import cProfile
    cProfile.run('your_function()')

**Line profiling:**

.. code-block:: bash

    pip install line_profiler
    # Use @profile decorator and run kernprof

**GPU profiling:**

For CUDA operations, use ``nvprof`` or ``nsys``

Common performance pitfalls
----------------------------

**Repeated network creation:** Cache networks instead of recreating

**Unnecessary data copies:** Many array operations create copies - use in-place operations when possible

**Type conversions:** Converting between array types is expensive - stick to one backend per workflow

**Reading data repeatedly:** Load data once, process multiple times

**Small batch sizes:** Vectorization benefits from larger batches

See also
--------

- :doc:`array_backend_design` - Understanding array backend choices
- :doc:`../howto/use_different_array_backends` - How to use different backends
