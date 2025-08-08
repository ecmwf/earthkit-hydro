Integration with the earthkit system
====================================

earthkit-hydro is the hydrological component of earthkit :cite:`earthkit`. It is designed to interplay with other earthkit components seamlessly, primarily via xarray integration.

Here is a simple example of using different earthkit packages together.

.. code-block:: python

    import earthkit.data as ekd
    import earthkit.hydro as ekh
    import earthkit.plots as ekp

    # specify some custom styles
    style = ekp.styles.Style(
        colors="Blues",
        levels=[0, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 4000],
        extend="max",
    )

    # load data and river network
    network = ekh.river_network.load("efas", "5")
    da = ekd.from_source(
        "sample",
        "R06a.nc",
    )[0].to_xarray()

    # compute upstream accumulation
    upstream_sum = ekh.upstream.sum(network, da)

    # plot result
    chart = ekp.Map()
    chart.quickplot(upstream_sum, style=style)
    chart.legend(label="{variable_name}")
    chart.title("Upstream precipitation at {time:%H:%M on %-d %B %Y}")
    chart.coastlines()
    chart.gridlines()
    chart.show()

.. image:: ../../images/earthkit_example.png
   :width: 100%
   :align: center
