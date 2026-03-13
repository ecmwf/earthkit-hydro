Installation and Getting Started
================================

Installing from PyPi
--------------------

It is recommended that you use the latest version of python3 and pip for
the installation of earthkit-hydro.

.. code:: bash

   pip install earthkit-hydro

To make use of the interoperable functionality you should ensure that
you have installed the *earthkit-data* dependency.


Import and use
--------------

.. code:: python

    import earthkit.hydro as ekh

    daily_mean = ekh.temporal.daily_mean(MY_DATA)
