Specifying locations
====================

Many functions are concerned with operations relating a subset of the entire river network i.e. a fixed number of locations. This can range from catchment averages, to distances etc.

The most convenient and common way to specify a gauge location is by its coordinates, typically latitude and longitude. The specified coordinates must match the river network coordinate system.

For the EFAS network (which uses regular lat/lon) we can specify via the lat/lon of points of interest.

.. code-block:: python

    locations = {
        "station1": (10, 10),
        "station2": (10, 10),
        "station3": (10, 10)
    }

    labelled_field = ekh.catchments.sum(network, field, locations)

However, for more performance, it is also possible to specify directly a grid index.

.. code-block:: python

    locations = [(10,10), (50, 30), (80, 70)]

    labelled_field = ekh.catchments.sum(network, field, locations)

Or, for maximum performance, it is possible to also specify node labels.

.. code-block:: python

    locations = [10, 5, 6]

    labelled_field = ekh.catchments.sum(network, field, locations)
