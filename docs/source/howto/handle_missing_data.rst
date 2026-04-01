Handling missing data
=====================

earthkit-hydro uses NumPy's ``np.nan`` convention for missing values. Any operation involving ``np.nan`` returns ``np.nan`` — this is known as NaN propagation. If you are coming from PCRaster, note that the missing value conventions differ; see :doc:`../concepts/missing_value_handling` for background.

.. code-block:: python

    import numpy as np
    import earthkit.hydro as ekh

    network = ekh.river_network.load("efas", "5")

    # Create a field with some missing values
    field = np.ones(network.shape)
    field[10:20, 15:25] = np.nan

NaN propagation
---------------

Most operations propagate NaN values naturally. If a grid cell or any of its upstream neighbours has a missing value, the result will be NaN:

.. code-block:: python

    result = ekh.upstream.sum(network, field)
    # result contains NaN wherever input or upstream input has NaN

This is typically the desired behaviour — if upstream data is unknown, the accumulated result is also unknown.

Filling missing values
----------------------

If NaN propagation is not appropriate for your use case, fill missing values before processing:

.. code-block:: python

    # Replace NaN with zero
    field_filled = np.nan_to_num(field, nan=0.0)
    result = ekh.upstream.sum(network, field_filled)

    # For xarray DataArrays
    field_xr = field_xr.fillna(0.0)
    result = ekh.upstream.sum(network, field_xr)

.. warning::

   Be careful when filling missing values. Replacing NaN with 0 assumes zero precipitation/runoff/etc., which may not be appropriate for all analyses.

Converting from other conventions
---------------------------------

Data that uses sentinel values (e.g. -999 or -9999) must be converted to NaN before processing:

.. code-block:: python

    field[field == -999] = np.nan
    result = ekh.upstream.sum(network, field)

See also
--------

- :doc:`../concepts/missing_value_handling` — Design rationale for the NaN convention
