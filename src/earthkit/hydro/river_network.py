from io import BytesIO
from urllib.request import urlopen

import joblib

from ._version import __version__ as ekh_version
from .readers import (
    cache,
    find_main_var,
    from_cama_nextxy,
    from_d8,
    import_earthkit_or_prompt_install,
)

# read in only up to second decimal point
# i.e. 0.1.dev90+gfdf4e33.d20250107 -> 0.1
ekh_version = ".".join(ekh_version.split(".")[:2])


@cache
def create(path, river_network_format, source):
    """Creates a river network from the given path, format, and source.

    Parameters
    ----------
    path : str
        The path to the river network data.
    river_network_format : str
        The format of the river network data.
        Supported formats are "precomputed", "cama", "pcr_d8", and "esri_d8".
    source : str
        The source of the river network data.
        For possible sources see:
        https://earthkit-data.readthedocs.io/en/latest/guide/sources.html.

    Returns
    -------
    object
        The river network object created from the given data.

    Raises
    ------
    ValueError
        If the river network format or source is unsupported.
    NotImplementedError
        If the river network format is "esri_d8".
    """
    if river_network_format == "precomputed":
        if source == "file":
            return joblib.load(path)
        elif source == "url":
            return joblib.load(BytesIO(urlopen(path).read()))
        else:
            raise ValueError(
                "Unsupported source for river network format"
                f"{river_network_format}: {source}."
            )
    elif river_network_format == "cama":
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        x, y = data.nextx.values, data.nexty.values
        return from_cama_nextxy(x, y)
    elif river_network_format == "pcr_d8" or river_network_format == "esri_d8":
        ekd = import_earthkit_or_prompt_install(river_network_format, source)
        data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
        var_name = find_main_var(data)
        return from_d8(data[var_name].values, river_network_format=river_network_format)
    else:
        raise ValueError(f"Unsupported river network format: {river_network_format}.")


def load(
    domain,
    river_network_version,
    data_source=(
        "https://github.com/ecmwf/earthkit-hydro-store/raw/refs/heads/main/"
        "{ekh_version}/{domain}/{river_network_version}/river_network.joblib"
    ),
    *args,
    **kwargs,
):
    """Load a precomputed river network from a named domain and
    river_network_version.

    Parameters
    ----------
    domain : str
        The domain of the river network. Supported domains are "efas", "glofas",
        "cama_15min", "cama_06min", "cama_05min", "cama_03min".
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
    earthkit.hydro.RiverNetwork
        The loaded river network.

    """
    uri = data_source.format(
        ekh_version=ekh_version,
        domain=domain,
        river_network_version=river_network_version,
    )
    return create(uri, "precomputed", "url", *args, **kwargs)


def available():
    """
    Prints the available precomputed networks.
    """

    print(
        "Available precomputed networks are:\n",
        '`ekh.river_network.load("efas", "5")`\n',
        '`ekh.river_network.load("efas", "4")`\n',
        '`ekh.river_network.load("glofas", "4")`\n',
        '`ekh.river_network.load("glofas", "3")`\n',
        '`ekh.river_network.load("cama_15min", "4")`\n',
        '`ekh.river_network.load("cama_06min", "4")`\n',
        '`ekh.river_network.load("cama_05min", "4")`\n',
        '`ekh.river_network.load("cama_03min", "4")`\n',
        '`ekh.river_network.load("hydrosheds_06min", "1")`\n',
        '`ekh.river_network.load("hydrosheds_05min", "1")`',
    )
