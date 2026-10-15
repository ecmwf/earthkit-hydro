# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from earthkit.hydro._utils.decorators import xarray

from .array import area as array_area
from .array import upstream_area as array_upstream_area


def area(river_network, return_type=None):
    r"""
    Returns the area of each node in the river network.

    Areas are given in SI units (:math:`\mathrm{m}^2`).

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    xarray object
        Array of areas in :math:`\mathrm{m}^2` for every river network node or gridcell, depending on `return_type`.
    """
    da = xarray(array_area)(river_network, return_type=return_type)
    da.name = "area"
    da.attrs["units"] = "m^2"
    return da


def upstream_area(river_network, return_type=None):
    r"""
    Computes the upstream area of each node in the river network.

    For each node in the river network, this function sums the areas of the node
    itself and of all nodes upstream of it, i.e. its contributing drainage area.

    The upstream area is defined as:

    .. math::
        :nowrap:

        \begin{align*}
        A_j &= a_j + \sum_{i \in \mathrm{Up}(j)} A_i
        \end{align*}

    where:

    - :math:`a_j` is the area of node :math:`j`,
    - :math:`\mathrm{Up}(j)` is the set of upstream nodes flowing into node :math:`j`,
    - :math:`A_j` is the upstream area at node :math:`j`.

    Accumulation proceeds in topological order from the sources to the sinks.
    Areas are given in SI units (:math:`\mathrm{m}^2`).

    Parameters
    ----------
    river_network : RiverNetwork
        A river network object.
    return_type : str, optional
        Either "masked", "gridded" or None. If None (default), uses `river_network.return_type`.

    Returns
    -------
    xarray object
        Array of upstream areas in :math:`\mathrm{m}^2` for every river network node or gridcell, depending on `return_type`.
    """
    da = xarray(array_upstream_area)(river_network, return_type=return_type)
    da.name = "upstream_area"
    da.attrs["units"] = "m^2"
    return da
