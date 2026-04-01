Missing value handling philosophy
==================================

This page explains earthkit-hydro's approach to missing values and why it differs from some other hydrological tools.

The NaN convention
------------------

earthkit-hydro uses NumPy's ``np.nan`` (Not a Number) to represent missing values. This is a deliberate design choice that aligns with the scientific Python ecosystem and provides clear, predictable behavior.

**Key principle:** Any operation involving a missing value returns a missing value.

This is known as **NaN propagation** and is fundamental to how earthkit-hydro handles uncertainty.

Why NaN propagation?
--------------------

NaN propagation ensures that missing or invalid data doesn't silently corrupt your results:

**Example scenario:** You're calculating upstream precipitation:

- Station A: 50 mm
- Station B: NaN (sensor failure)
- Station C: 30 mm

If station B drains to the outlet:

.. code-block:: python

    upstream_sum = ekh.upstream.sum(network, precipitation)
    # Result at outlet = NaN

The outlet shows NaN because we honestly don't know the total—we're missing data from station B.

**Alternative (dangerous) approach:** Treating NaN as zero would give:

.. code-block:: python

    # If we wrongly treated NaN as 0
    # Result at outlet = 50 + 0 + 30 = 80 mm  <- WRONG!

This incorrect result (80 mm) could mislead decisions. The NaN result correctly signals "we don't have complete information."

Comparison with PCRaster
-------------------------

PCRaster, a widely-used hydrological tool, handles missing values differently:

**PCRaster approach:**

- Uses a special missing value marker (MV)
- Some operations skip missing values
- Some operations propagate missing values
- Behavior varies by operation

**earthkit-hydro approach:**

- Always uses ``np.nan``
- Always propagates missing values
- Consistent behavior across all operations
- Explicit handling required

**Why the difference?**

PCRaster was designed when skipping missing values was common practice. earthkit-hydro prioritizes:

- **Transparency:** NaN propagation makes missing data visible
- **Safety:** Prevents silent errors from ignored missing data
- **Consistency:** Same rules for all operations
- **Ecosystem:** Compatible with pandas, xarray, NumPy conventions

When this matters
-----------------

The distinction is most important for:

**Accumulation operations:**

If you have missing precipitation data upstream, the downstream total should be NaN (unknown), not the sum of available stations.

**Catchment statistics:**

If part of a catchment has missing data, the catchment mean should be NaN unless you explicitly decide how to handle gaps.

**Time series analysis:**

Missing values at any time step should propagate, alerting you to data quality issues.

How to handle missing values
-----------------------------

earthkit-hydro's approach requires you to make explicit choices about missing data:

**Option 1: Fill with a value**

.. code-block:: python

    import numpy as np

    # Replace NaN with zero (assumes missing = zero)
    field_filled = np.nan_to_num(field, nan=0.0)
    result = ekh.upstream.sum(network, field_filled)

**Option 2: Interpolate**

.. code-block:: python

    # Spatially interpolate gaps
    field_interpolated = interpolate_missing(field)
    result = ekh.upstream.sum(network, field_interpolated)

**Option 3: Work with NaN**

.. code-block:: python

    # Accept NaN in results, handle downstream
    result = ekh.upstream.sum(network, field)
    # Check which locations have complete data
    valid_results = ~np.isnan(result)

**Option 4: Skip missing regions**

.. code-block:: python

    # Only process where data is complete
    mask = ~np.isnan(field)
    result = ekh.upstream.sum(network, field, mask=mask)

Each choice has implications—earthkit-hydro forces you to think about them.

Benefits of explicit handling
------------------------------

**Prevents silent errors:**

You can't accidentally use results based on incomplete data without realizing it.

**Encourages data quality awareness:**

NaN propagation makes you aware of your data's completeness.

**Compatibility:**

Works seamlessly with pandas, xarray, and other scientific Python tools that use the same convention.

**Reproducibility:**

Explicit missing value handling makes analyses more reproducible and easier to understand.

When you might prefer PCRaster's approach
------------------------------------------

PCRaster's approach can be more convenient when:

- You have many small gaps you want to ignore
- You're replicating historical analyses that used PCRaster
- You want operations to "work around" missing data automatically

However, this convenience comes at the cost of transparency and potential silent errors.

Migration from PCRaster
-----------------------

If you're migrating from PCRaster:

**Step 1: Identify missing value handling**

Review your PCRaster code for operations that skip missing values.

**Step 2: Decide on explicit strategy**

Choose how to handle each case:

- Fill with zero (if appropriate)
- Interpolate (if justified)
- Keep as NaN (if uncertainty acceptable)

**Step 3: Implement in earthkit-hydro**

Use NumPy/xarray functions to implement your strategy explicitly.

**Example:**

.. code-block:: python

    # PCRaster (implicit missing value handling)
    result = accuflux(ldd, field)  # May skip missing values

    # earthkit-hydro (explicit handling)
    field_filled = field.fillna(0)  # Explicit choice
    result = ekh.upstream.sum(network, field_filled)

Best practices
--------------

1. **Document your choices:** Note how you handle missing values in comments/documentation

2. **Validate results:** Check ``np.sum(np.isnan(result))`` to see how much data is missing

3. **Propagate metadata:** Use xarray to track data quality flags alongside values

4. **Consider uncertainty:** NaN results indicate uncertainty—report this in your analysis

5. **Be consistent:** Use the same missing value strategy throughout an analysis

See also
--------

- :doc:`../howto/handle_missing_data` - Practical guide to working with gaps
- :doc:`pcraster_compatibility` - Comparison with PCRaster
