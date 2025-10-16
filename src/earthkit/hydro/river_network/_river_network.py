from io import BytesIO
from urllib.request import urlopen

import joblib

from earthkit.hydro._readers import (
    find_main_var,
    from_cama_nextxy,
    from_d8,
    from_grit,
    import_earthkit_or_prompt_install,
)
from earthkit.hydro._utils.coords import get_core_grid_dims
from earthkit.hydro._utils.readers import from_file
from earthkit.hydro._version import __version__ as ekh_version
from earthkit.hydro.data_structures._network import RiverNetwork

from ._cache import cache

# read in major version
# if dev version, try add +1 to major version
# i.e. 0.1.dev90+gfdf4e33.d20250107 -> 1
# i.e. 0.1.0 -> 0
ekh_version = (
    int(ekh_version.split(".")[0]) + 1
    if "dev" in ekh_version
    else int(ekh_version.split(".")[0])
)


@cache
def create(
    path,
    river_network_format,
    source="file",
    use_cache=True,
    cache_dir=None,
    cache_fname="{ekh_version}_{hash}.joblib",
    cache_compression=1,
):
    """
    Creates a river network from the given path, format, and source.

    Parameters
    ----------
    path : str
        The path to the river network data. All common file formats are supported such
        as netCDF, GRIB, GeoTIFF, zarr, etc.
    river_network_format : str
        The format of the river network data.
        Supported formats are "precomputed", "cama", "pcr_d8", "esri_d8", "grit"
        and "merit_d8".
    source : str
        The source of the river network data. Default is `'file'`.
        For possible sources see:
        https://earthkit-data.readthedocs.io/en/latest/guide/sources.html.
    use_cache : bool, optional
        Whether to cache the loaded/created river network for quicker reloading. Default is True.
    cache_dir : str, optional
        Where to store the cached river networks. Default is None, which uses `tempfile.mkdtemp(suffix="_earthkit_hydro")`.
    cache_fname : str, optional
        A string template for the cache filename convention.
    cache_compression : int, optional
        A compression factor for the cached files.

    Returns
    -------
    RiverNetwork
        The river network object created from the given data.
    """
    if river_network_format == "precomputed":
        if source == "file":
            river_network_storage = joblib.load(path)
        elif source == "url":
            with urlopen(path) as response:
                river_network_storage = joblib.load(BytesIO(response.read()))
        else:
            raise ValueError(
                "Unsupported source for river network format"
                f"{river_network_format}: {source}."
            )
    elif river_network_format == "cama":
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        x, y = data.nextx.values, data.nexty.values
        river_network_storage = from_cama_nextxy(x, y)
        coord1, coord2 = get_core_grid_dims(data)
        river_network_storage.coords = {
            coord1: data[coord1].values,
            coord2: data[coord2].values,
        }
    elif (
        river_network_format == "pcr_d8"
        or river_network_format == "esri_d8"
        or river_network_format == "merit_d8"
    ):
        if path.endswith(".map"):
            data = from_file(path, mask=False)
            river_network_storage = from_d8(
                data, river_network_format=river_network_format
            )
            # coords not available
        else:
            ekd = import_earthkit_or_prompt_install(river_network_format, source)
            data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
            coord1, coord2 = get_core_grid_dims(data)
            var_name = find_main_var(data)
            river_network_storage = from_d8(
                data[var_name].values, river_network_format=river_network_format
            )
            river_network_storage.coords = {
                coord1: data[coord1].values,
                coord2: data[coord2].values,
            }
    elif river_network_format == "grit":
        assert path.endswith(".gpkg")
        river_network_storage = from_grit(path)
    else:
        raise ValueError(f"Unsupported river network format: {river_network_format}.")

    return RiverNetwork(river_network_storage)


def load(
    domain,
    river_network_version,
    data_source=(
        "https://sites.ecmwf.int/repository/earthkit-hydro/"
        "{ekh_version}/{domain}/{river_network_version}/river_network.joblib"
    ),
    *args,
    **kwargs,
):
    """
    Load a precomputed river network from a named domain and
    river_network_version.

    Supported networks are as follows:

    +----------------------+-----------+---------------------+----------------------------+----------------+
    | `domain`             | `version` | Details             | Note                       | Attribution    |
    +======================+===========+=====================+============================+================+
    | "efas"               | "5"       | 1arcmin European    |                            | [1]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "efas"               | "4"       | 5km European        | Smaller domain than v5     | [1]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "glofas"             | "4"       | 3arcmin global      | 60° South to 90° North     | [2]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "glofas"             | "3"       | 6arcmin global      | 60° South to 90° North     | [2]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "cama_01min"         | "4"       | 3arcmin global      |                            | [3]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "cama_03min"         | "4"       | 3arcmin global      |                            | [3]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "cama_05min"         | "4"       | 5arcmin global      |                            | [3]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "cama_06min"         | "4"       | 6arcmin global      |                            | [3]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "cama_15min"         | "4"       | 15arcmin global     |                            | [3]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "hydrosheds_30sec"   | "1"       | 30arcsec global     | 56° South to 84° North     | [4]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "hydrosheds_05min"   | "1"       | 5arcmin global      | 56° South to 84° North     | [4]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "hydrosheds_06min"   | "1"       | 6arcmin global      | 56° South to 84° North     | [4]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+
    | "grit"               | "1"       | 30m global (vector) | segments network           | [5]_           |
    +----------------------+-----------+---------------------+----------------------------+----------------+


    Parameters
    ----------
    domain : str
        The domain of the river network.
    river_network_version : str
        The version of the river network on the specified domain.
    data_source : str, optional
        The data source URL template for the river network.
    *args : tuple
        Additional positional arguments to pass to `create_river_network`.
    **kwargs : dict
        Additional keyword arguments to pass to `create_river_network`.

    Returns
    -------
    RiverNetwork
        The loaded river network.

    References
    ----------
    .. [1] Choulga, Margarita; Moschini, Francesca; Mazzetti, Cinzia; Grimaldi, Stefania; Disperati, Juliana; Beck, Hylke; Salamon, Peter; Prudhomme, Christel (2023): LISFLOOD static and parameter maps for Europe. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/f572c443-7466-4adf-87aa-c0847a169f23
    .. [2] Choulga, Margarita; Moschini, Francesca; Mazzetti, Cinzia; Disperati, Juliana; Grimaldi, Stefania; Beck, Hylke; Salamon, Peter; Prudhomme, Christel (2023): LISFLOOD static and parameter maps for GloFAS. European Commission, Joint Research Centre (JRC) [Dataset] PID: http://data.europa.eu/89h/68050d73-9c06-499c-a441-dc5053cb0c86
    .. [3] Yamazaki, Dai; Ikeshima, Daiki; Sosa, Jeison; Bates, Paul D.; Allen, George H.; Pavelsky, Tamlin M. (2019): MERIT Hydro: A high-resolution global hydrography map based on latest topography datasets. Water Resources Research, vol.55, pp.5053-5073, 2019, DOI: 10.1029/2019WR024873
    .. [4] Lehner, Bernhard; Verdin, Kristine; Jarvis, Andy (2008): New global hydrography derived from spaceborne elevation data. Eos, Transactions, 89(10): 93-94. Data available at https://www.hydrosheds.org.
    .. [5] Wortmann, Michel; Slater, Louise; Hawker, Laurence; Liu, Yinxue; Neal, Jeffrey; Zhang, Boen; Schwenk, Jon; Allen, George H.; Ashworth, Philip; Boothroyd, Richard; Cloke, Hannah; Delorme, Pauline; Gebrechorkos, Solomon H.; Griffith, Helen; Leyland, Julian; McLelland, Stuart; Nicholas, Andrew P.; Sambrook-Smith, Gregory; Vahidi, Elham; Parsons, Daniel; Darby, Stephen E. (2025). Global River Topology (GRIT): A bifurcating river hydrography. Water Resources Research, 61(5), DOI: 10.1029/2024WR038308
    """

    try:
        uri = data_source.format(
            ekh_version=ekh_version,
            domain=domain,
            river_network_version=river_network_version,
        )
        network = create(uri, "precomputed", "url", *args, **kwargs)
    except Exception:
        uri = data_source.format(
            ekh_version=ekh_version - 1,
            domain=domain,
            river_network_version=river_network_version,
        )
        network = create(uri, "precomputed", "url", *args, **kwargs)

    return network


def available(
    data_source="https://sites.ecmwf.int/repository/earthkit-hydro/available.txt",
):
    """
    Prints the available precomputed networks.

    Parameters
    ----------
    data_source : str, optional
        Base URI to read available networks from.
    """

    with urlopen(data_source) as response:
        html = response.read()

    print("Available precomputed networks are:\n", html.decode("utf-8"))
