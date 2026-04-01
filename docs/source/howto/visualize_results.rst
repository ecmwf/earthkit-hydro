Visualising results
===================

earthkit-hydro returns standard NumPy arrays or xarray DataArrays, so any Python plotting library works.

Basic matplotlib
----------------

.. code-block:: python

    import earthkit.hydro as ekh
    import matplotlib.pyplot as plt
    import numpy as np

    network = ekh.river_network.load("efas", "5")
    field = np.ones(network.shape)
    result = ekh.upstream.sum(network, field)

    plt.figure(figsize=(12, 8))
    plt.imshow(result, cmap='viridis')
    plt.colorbar(label='Upstream accumulation')
    plt.show()

xarray plotting
---------------

If the result is an xarray DataArray, coordinates are handled automatically:

.. code-block:: python

    result.plot.contourf(levels=20, cmap='Blues', figsize=(12, 8))
    plt.show()

Cartopy maps
------------

For geographic projections with coastlines:

.. code-block:: python

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(lons, lats, result, levels=20, cmap='Blues', transform=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    plt.colorbar(im, ax=ax, label='Upstream accumulation', orientation='horizontal', pad=0.05)
    plt.show()

earthkit-plots
--------------

For quick, consistent visualisation using the earthkit ecosystem:

.. code-block:: python

    import earthkit.plots as ekp

    style = ekp.styles.Style(
        colors="Blues",
        levels=[0, 10, 50, 100, 500, 1000, 2000, 5000, 10000],
        extend="max",
    )

    chart = ekp.Map()
    chart.quickplot(result, style=style)
    chart.coastlines()
    chart.show()

Catchment boundaries
--------------------

Plot catchment labels with a discrete colormap:

.. code-block:: python

    catchments = ekh.catchments.find(network, locations)

    plt.figure(figsize=(12, 8))
    plt.imshow(catchments, cmap='tab20', interpolation='nearest')
    plt.colorbar(label='Catchment ID')
    plt.show()
