# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from earthkit.hydro._backends.numpy_backend import NumPyBackend
from earthkit.hydro._utils.decorators import mask
from earthkit.hydro.upstream import array

def _area(xp, river_network, area):
    return river_network._storage.area

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
    array-like
        Array of areas in :math:`\mathrm{m}^2` for every river network node or gridcell, depending on `return_type`.
    """
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")
    return_type = river_network.return_type if return_type is None else return_type
    if return_type not in ["gridded", "masked"]:
        raise ValueError("return_type must be either 'gridded' or 'masked'.")
    func = mask(return_type == "gridded")(_area)
    return func(river_network=river_network, xp=NumPyBackend(), field=river_network._storage.area)


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
    array-like
        Array of upstream areas in :math:`\mathrm{m}^2` for every river network node or gridcell, depending on `return_type`.
    """
    if river_network._storage.area is None:
        raise ValueError("River network pixel area is not available.")
    return array.sum(river_network, field=river_network._storage.area, return_type=return_type)
