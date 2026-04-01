Topological sorting in river networks
=====================================

Topological sorting is a fundamental operation in earthkit-hydro that enables efficient hydrological calculations.
This page explains what it is, why it's necessary, and its performance implications.

What is topological sorting?
-----------------------------

Topological sorting arranges river network nodes in an order where upstream nodes always come before downstream nodes.

**Example:** For a simple network:

.. code-block:: text

    Source1 → Node1 ↘
                      Node3 → Outlet
    Source2 → Node2 ↗

A valid topological order might be:

.. code-block:: python

    [Source1, Source2, Node1, Node2, Node3, Outlet]

**Key property:** When processing nodes in this order, all upstream contributions are available before processing downstream nodes.

Why is it necessary?
--------------------

Topological sorting enables **single-pass algorithms** for accumulation operations.

**Without topological sorting:**

.. code-block:: python

    # Naive approach: iterate until convergence
    converged = False
    while not converged:
        for each_node:
            sum_upstream_values()
        check_convergence()
    # Slow! May require many iterations

**With topological sorting:**

.. code-block:: python

    # Efficient approach: single pass
    for node in topologically_sorted_order:
        sum_upstream_values()
    # Fast! Only one pass needed

For large networks (millions of nodes), this difference is dramatic:

- Without sorting: Seconds to minutes
- With sorting: Milliseconds

How earthkit-hydro uses it
---------------------------

When you create or load a river network, earthkit-hydro:

1. **Analyzes network structure** - Identifies all upstream/downstream relationships
2. **Sorts nodes** - Arranges them in valid topological order
3. **Caches the ordering** - Stores for reuse in operations

This one-time computation cost enables fast repeated operations.

The sorting algorithm
---------------------

earthkit-hydro uses a depth-first search (DFS) based algorithm:

1. Start from source nodes (no upstream)
2. Recursively visit downstream nodes
3. Add nodes to sorted list after visiting all upstream

**Time complexity:** O(N + E) where N = nodes, E = edges

**Space complexity:** O(N) for storing the sorted order

For typical river networks:

- Small networks (< 10k nodes): Milliseconds
- Medium networks (100k-1M nodes): Seconds
- Large networks (> 1M nodes): 10s of seconds

Why creating networks is slow
------------------------------

When you call:

.. code-block:: python

    network = ekh.river_network.create(path, format, source)

The process involves:

1. **Load data** (fast - seconds)
2. **Parse flow directions** (fast - seconds)
3. **Build graph structure** (medium - seconds)
4. **Topological sorting** (slow for large networks - minutes)
5. **Validate and optimize** (fast - seconds)

**Step 4 is the bottleneck** for large networks.

That's why pre-computed networks are much faster:

.. code-block:: python

    # Load pre-computed (sorted already)
    network = ekh.river_network.load("efas", "5")  # Seconds

    # vs create from scratch
    network = ekh.river_network.create(...)  # Minutes

Why not sort on-demand?
-----------------------

**Question:** Why not skip sorting and sort only when needed?

**Answer:** Because nearly every operation requires it:

- ``upstream.sum()`` - needs topological order
- ``downstream.sum()`` - needs reverse topological order
- ``catchments.find()`` - needs topological order
- ``distance.*`` - needs topological order
- ``length.*`` - needs topological order

Sorting once upfront is much more efficient than sorting repeatedly or using slower algorithms.

Cyclic networks
---------------

Topological sorting only works for **directed acyclic graphs** (DAGs). River networks should be acyclic (water doesn't flow in circles), but data errors can create cycles.

If earthkit-hydro detects a cycle:

.. code-block:: python

    # This will raise an error
    network = ekh.river_network.create(path_with_cycle, format, source)
    # Error: Cyclic network detected at node(s) [...]

**Common causes of cycles:**

- Data entry errors
- Incorrect flow direction encoding
- Corrupted network files
- Bidirectional flow specifications

**How to fix:**

1. Fix source data if possible
2. Use ``check_for_cycles()`` to identify problem nodes
3. Manually correct or remove problematic connections

Bifurcations and sorting
-------------------------

Bifurcations (one node flowing to multiple downstream nodes) don't break topological sorting.
A valid ordering still exists, with the branching node before all its downstream nodes.

**Example with bifurcation:**

.. code-block:: text

    Source → Node1 → Node2 ↗ Outlet1
                            ↘ Outlet2

Valid order: ``[Source, Node1, Node2, Outlet1, Outlet2]``

(The order of Outlet1 and Outlet2 doesn't matter—they're independent after the bifurcation.)

Performance tips
----------------

**1. Cache networks**

.. code-block:: python

    # Create once, export
    network = ekh.river_network.create(path, format, source)
    network.export("cached_network.joblib")

    # Reuse in all subsequent analyses
    network = ekh.river_network.create("cached_network.joblib", "precomputed")

**2. Use pre-computed networks**

.. code-block:: python

    # Prefer this
    network = ekh.river_network.load("efas", "5")

    # Over this
    network = ekh.river_network.create("efas_raw.nc", "pcr_d8", "file")

**3. Subset before sorting**

If you only need a small region:

.. code-block:: python

    # Load full network (already sorted)
    full_network = ekh.river_network.load("efas", "5")

    # Create subnetwork (no re-sorting needed)
    region_mask = (lats > 40) & (lats < 50)
    subnetwork = ekh.subnetwork.from_mask(full_network, region_mask)

This avoids sorting the full network from scratch.

Advanced: Multiple topological orderings
-----------------------------------------

A river network can have multiple valid topological orderings.

**Example:**

.. code-block:: text

    Source1 → Node1 ↘
                      Outlet
    Source2 → Node2 ↗

Valid orderings:

- ``[Source1, Node1, Source2, Node2, Outlet]``
- ``[Source1, Source2, Node1, Node2, Outlet]``
- ``[Source2, Node2, Source1, Node1, Outlet]``

All are correct! earthkit-hydro chooses one deterministically for consistency.

This means:

- Results are reproducible
- Same network always gives same ordering
- But the specific order is implementation-dependent

Implementation details
----------------------

earthkit-hydro's implementation includes optimizations:

1. **Parallel sorting** for independent sub-basins
2. **Incremental updating** for subnetworks
3. **Cached grouping** for identical topological levels

For most users, these details don't matter—just know that sorting is fast and reliable.

Comparison with other tools
----------------------------

**PCRaster:** Also uses topological sorting, but less transparent about it

**ArcGIS Hydro:** Uses different algorithms (doesn't expose topological ordering)

**TauDEM:** Creates topological ordering implicitly during processing

**earthkit-hydro:** Makes topological sorting explicit and reusable across operations

See also
--------

- :doc:`../howto/load_river_network` - Loading and caching networks
- :doc:`../explanation/river_network_concepts` - River network representation
- :doc:`../explanation/performance_considerations` - Optimization strategies
- :doc:`../howto/optimize_performance` - Practical performance tips
