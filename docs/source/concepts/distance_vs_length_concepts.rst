Distance vs. length concepts
=============================

Understanding the distinction between distance and length is fundamental to working with river networks in earthkit-hydro.
While these terms are often used interchangeably in everyday language, they represent fundamentally different concepts in hydrological network analysis.

Why this distinction matters
-----------------------------

In many hydrological tools, distance and length calculations are conflated, which can lead to:

- Incorrect routing time estimates
- Errors in distance-decay calculations
- Confusion at confluences and bifurcations
- Inappropriate weighting schemes

earthkit-hydro makes this distinction explicit to enable more accurate and flexible analysis.

Node properties vs. edge properties
------------------------------------

The fundamental difference lies in what is being measured:

**Lengths are node properties**

A length is associated with a grid cell or graph node itself. It represents the length of river channel within that cell.

- One length value per node
- Represents channel length within the cell
- Remains constant even at confluences
- Used for: channel residence time, friction calculations, node-based weighting

**Distances are edge properties**

A distance is associated with the connection (edge) between two nodes. It represents the distance traveled along the flow path from one node to another.

- One distance value per edge (connection between nodes)
- Represents distance between cell centers (or similar)
- Can differ for each branch at confluences/bifurcations
- Used for: travel distance calculations, path finding, edge-based weighting

Visual explanation
------------------

Consider this simple river network:

.. image:: ../../images/distance_length.png
   :width: 500px
   :align: center

In the highlighted segment:

- **Length = 3** (sum of channel lengths within the 3 cells)
- **Distance = 2** (number of connections/edges traveled)

Even for a straight channel with uniform cells, these values differ because:

- Length accounts for the actual channel path within each cell
- Distance counts the connections between cell centers

Implications at confluences
----------------------------

The distinction becomes especially important at confluences:

**Scenario:** Two tributaries meet at a confluence node.

- The confluence node has **one length** (the channel length within that cell)
- But there are **multiple distances** (one for each incoming branch)

This means:

- A parcel of water traveling down tributary A experiences distance_A to the confluence
- A parcel from tributary B experiences distance_B to the confluence
- Both experience the same length when passing through the confluence cell

**Why this matters:** If you're calculating travel time with distance-dependent decay (e.g., for pollutant attenuation), you need edge-based distances to correctly account for different paths to the confluence.

When to use each
----------------

**Use lengths when:**

- Calculating channel residence time (velocity × length)
- Applying friction or roughness coefficients
- Computing channel storage capacity
- Weighting by actual river channel amount

**Use distances when:**

- Finding shortest/longest path between points
- Calculating travel distance through the network
- Implementing distance-decay functions
- Routing based on connection topology

Mathematical formulation
------------------------

For a path through nodes :math:`n_1, n_2, ..., n_k`:

.. math::

   \text{Total length} = \sum_{i=1}^{k} L(n_i)

   \text{Total distance} = \sum_{i=1}^{k-1} D(n_i, n_{i+1})

Where:

- :math:`L(n_i)` is the length property of node :math:`i`
- :math:`D(n_i, n_{i+1})` is the distance property of the edge from node :math:`i` to :math:`i+1`

Note that length is summed over :math:`k` nodes, while distance is summed over :math:`k-1` edges.

Relationship to graph theory
-----------------------------

This distinction aligns with standard graph theory terminology:

- **Node weights** (lengths) = properties of vertices
- **Edge weights** (distances) = properties of edges

River networks are **edge-weighted directed graphs** where both node and edge properties matter for hydrological calculations.

Historical context
------------------

Many hydrological tools (including some versions of PCRaster) have traditionally used "distance" to refer to what is technically a length measurement, or conflated the two concepts. This simplification works for some analyses but breaks down for:

- Complex routing schemes
- Networks with varying cell sizes
- Bifurcation handling
- Distance-dependent processes

earthkit-hydro's explicit distinction enables more sophisticated and accurate analyses while remaining clear about what is being calculated.

Practical example
-----------------

**Problem:** Calculate pollutant concentration downstream, with 10% attenuation per unit distance traveled.

**Wrong approach:** Use lengths

.. code-block:: python

    # This uses channel length within cells
    # Attenuation will be too high (counts length within destination cell)
    result = ekh.upstream.sum(network, concentration * 0.9**lengths)

**Correct approach:** Use distances

.. code-block:: python

    # This uses distance between cells
    # Attenuation correctly represents travel distance
    result = ekh.upstream.sum(network, concentration * 0.9**distances)

The difference can be significant, especially in networks with variable cell sizes or long channels within cells.

Implementation in earthkit-hydro
---------------------------------

earthkit-hydro provides separate APIs for distances and lengths:

- ``ekh.distance.*`` - for edge-based distance calculations
- ``ekh.length.*`` - for node-based length calculations

Both support:

- Minimum and maximum (shortest/longest path)
- Upstream and downstream directions
- Custom weighting

See the how-to guides for practical usage examples.

See also
--------

- :doc:`../howto/specify_locations` - How to specify locations for distance/length calculations
- :doc:`../howto/calculate_river_distances` - How to compute distances between points
- :doc:`river_network_concepts` - Understanding river network representation
