# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from earthkit.hydro.data_structures import RiverNetwork
from earthkit.hydro.data_structures._network_storage import RiverNetworkStorage

from ._formats import FORMATS


def export(
    river_network: RiverNetworkStorage | RiverNetwork,
    path: str,
    river_network_format: str = "precomputed",
    compression=1,
):
    """
    Export a river network to a local file.

    .. note::
        Exporting to precomputed format is highly recommended for efficiency reasons.
        Other river network formats should only be used if compatibility is required with other tooling.

    .. note::
        For formats other than precomputed, only exporting as netcdf is currently supported.

    .. warning::
        The cama format has two different sink representations, one for inland sinks (-10), and one for coastal sinks (-9).
        There is only one sink representation in earthkit-hydro, so for cama format exports all sinks are exported as coastal sinks (-9).
        This does not change any results with earthkit-hydro, but be aware when using other tools.

    Parameters
    ----------
    river_network : RiverNetworkStorage | RiverNetwork
        The river network to export.
    path : str
        Where to export the river network.
    river_network_format : str
        The format of the river network data.
        Currently supported formats are "pcr_d8", "esri_d8"
        and "merit_d8".
    compression : int
        The compression factor to use for the saved file. Only applied if river_network_format is precomputed.

    Returns
    -------
    None. Writes the river network to a local file at `path`.
    """
    fmt = FORMATS.get(river_network_format)
    if fmt is None or not hasattr(fmt, "export_to"):
        raise ValueError(f"Exporting river network to format {river_network_format} is not currently supported.")

    if isinstance(river_network, RiverNetwork) and river_network.array_backend != "numpy":
        raise ValueError("Exporting for non-numpy backend not supported.")

    river_network_storage = river_network if isinstance(river_network, RiverNetworkStorage) else river_network._storage

    fmt.export_to(river_network_storage, path, compression)
