earthkit-hydro
==============

.. important::

    This software is **Incubating** and subject to ECMWF's guidelines on `Software Maturity <https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity>`_.

**earthkit-hydro** is a Python library for common hydrological functions.
It is the hydrological component of earthkit :cite:`earthkit`.

**Main Features**

.. raw:: html

   <div style="float: left; margin-right: 30px;">

.. https://agupubs.onlinelibrary.wiley.com/cms/asset/e10b31b2-7a5c-498d-bb27-49966867e6a8/wrcr70124-fig-0002-m.jpg
.. figure:: ../images/glofas.png
   :width: 300px

   *Adapted from:* :cite:`doc_figure`

.. raw:: html

   </div>

- Catchment delineation
- Catchment-based statistics
- Directional flow-based accumulations
- River network distance calculations
- Upstream/downstream field propagation
- Bifurcation handling
- Custom weighting and decay support

.. raw:: html

   <br style="clear: both">

.. image:: ../images/array_backends_with_xr.png
   :width: 300px
   :align: right

- Support for PCRaster, CaMa-Flood, HydroSHEDS, MERIT-Hydro and GRIT river network formats
- Compatible with major array-backends: xarray, numpy, cupy, torch, jax, mlx and tensorflow
- GPU support
- Differentiable operations suitable for machine learning

.. grid:: 1
   :gutter: 2

   .. grid-item-card:: Installation and Getting Started
      :img-top: _static/rocket.svg
      :link: getting-started
      :link-type: doc
      :class-card: sd-shadow-sm

      New to earthkit-hydro? Start here with installation and a quick overview.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Tutorials
      :img-top: _static/book.svg
      :link: tutorials/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Step-by-step guides to learn earthkit-hydro.

   .. grid-item-card:: How-tos
      :img-top: _static/tool.svg
      :link: how-tos/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Practical recipes for common tasks.

   .. grid-item-card:: Concepts and Explanations
      :img-top: _static/bulb.svg
      :link: userguide/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Understand the core ideas behind earthkit-hydro.

   .. grid-item-card:: API Reference Guide
      :img-top: _static/brackets-contain.svg
      :link: autodocs/earthkit.hydro
      :link-type: doc
      :class-card: sd-shadow-sm

      Detailed documentation of all functions and classes.



**Support**

Have a feature request or found a bug? Feel free to open an
`issue <https://github.com/ecmwf/earthkit-hydro/issues/new/choose>`_.


.. toctree::
   :caption: User guide
   :maxdepth: 2
   :hidden:

   getting-started
   tutorials/index
   userguide/index
   autodocs/earthkit.hydro


.. toctree::
   :caption: Developer guide
   :maxdepth: 2
   :hidden:

   contributing


.. toctree::
   :maxdepth: 2
   :caption: Extras
   :hidden:

   references
   genindex


.. toctree::
   :maxdepth: 2
   :caption: Related projects
   :hidden:

   earthkit <https://earthkit.readthedocs.io/en/latest>
   earthkit-data <https://earthkit-data.readthedocs.io/en/latest>
   earthkit-plots <https://earthkit-plots.readthedocs.io/en/latest>
   earthkit-meteo <https://earthkit-meteo.readthedocs.io/en/latest>
   earthkit-transforms <https://earthkit-transforms.readthedocs.io/en/latest>
