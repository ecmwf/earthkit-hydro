Integration with the earthkit system
====================================

TODO: figure out a good example that actually works right now

earthkit-hydro is the hydrological component of earthkit :cite:`earthkit`. It is designed to interplay with other earthkit components seamlessly, primarily via xarray integration.

Here is a contrived example of using different earthkit packages together to compare two simulations for a catchment of interest.
Specifically, the code below:

1. Loads two simulations using earthkit-data
2. Regrids them onto the EFAS network using earthkit-regrid
3. Computes the CRPS score using earthkit-meteo
4. Computes the monthly average CRPS score using earthkit-transforms
5. Finds the average CRPS per catchment using earthkit-hydro
6. Plots the results using earthkit-plots

.. code-block:: python

    import earthkit.data as ekd
    import earthkit.hydro as ekh
    import earthkit.meteo as ekm
    import earthkit.regrid as ekr
    import earthkit.plots as ekp
    from earthkit.transforms import aggregate

    network = ekh.river_network.load("efas", "5")

    ds_A = ekd.from_source("file", "simulation.nc").to_xarray()
    ds_B = ekd.from_source("file", "reanalysis.nc").to_xarray()

    ds_A = ekr.regrid()
    ds_B = ekr.regrid()

    crps = ekm.score.crps(ds_A, ds_B)

    monthly_catchment_mean = aggregate.climatology.monthly_mean(ds)

    catchment_mean = ekh.catchments.mean(network, ds, weights)

    ekp.quickplot(catchment_mean)
