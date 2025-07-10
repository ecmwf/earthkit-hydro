import json
from io import BytesIO
from urllib.request import Request, urlopen

import joblib

from earthkit.hydro._version import __version__ as ekh_version
from earthkit.hydro.data_structures.network import RiverNetwork
from earthkit.hydro.readers import (  # cache, from_grit,
    find_main_var,
    from_cama_nextxy,
    from_d8,
    import_earthkit_or_prompt_install,
)
from earthkit.hydro.utils.readers import from_file

from .cache import cache

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
        Supported formats are "precomputed", "cama", "pcr_d8", "esri_d8"
        and "merit_d8".
    source : str
        The source of the river network data.
        For possible sources see:
        https://earthkit-data.readthedocs.io/en/latest/guide/sources.html.

    Returns
    -------
    earthkit.hydro.network.RiverNetwork
        The river network object created from the given data.

    """
    if river_network_format == "precomputed":
        if source == "file":
            river_network_storage = joblib.load(path)
        elif source == "url":
            river_network_storage = joblib.load(BytesIO(urlopen(path).read()))
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
    elif (
        river_network_format == "pcr_d8"
        or river_network_format == "esri_d8"
        or river_network_format == "merit_d8"
    ):
        if path.endswith(".map"):
            data = from_file(path, mask=False)
        else:
            ekd = import_earthkit_or_prompt_install(river_network_format, source)
            data = ekd.from_source(source, path).to_xarray(mask_and_scale=False)
            var_name = find_main_var(data)
            data = data[var_name].values
        river_network_storage = from_d8(data, river_network_format=river_network_format)
    # elif river_network_format == "grit":
    #     assert path.endswith(".gpkg")
    #     river_network_storage = from_grit(path)
    else:
        raise ValueError(f"Unsupported river network format: {river_network_format}.")

    return RiverNetwork(river_network_storage)


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
    earthkit.hydro.network.RiverNetwork
        The loaded river network.

    """
    uri = data_source.format(
        ekh_version=ekh_version,
        domain=domain,
        river_network_version=river_network_version,
    )
    network = create(uri, "precomputed", "url", *args, **kwargs)

    return network


def available(
    data_source="https://api.github.com/repos/ecmwf/earthkit-hydro-store/git",
    token=None,
):
    """
    Prints the available precomputed networks.

    Parameters
    ----------
    data_source : str, optional
        Base github URI to query from.
    token : str, optional
        Github access token.
    """

    base_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "python-urllib-client",
    }
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    def github_api_request(url):
        req = Request(url, headers=base_headers)
        with urlopen(req) as response:
            if response.status != 200:
                raise Exception(f"GitHub API error {response.status}")
            return json.loads(response.read().decode())

    # get commit sha for main branch
    commit_sha = github_api_request(f"{data_source}/refs/heads/main")["object"]["sha"]

    # get entire tree
    tree_sha = github_api_request(f"{data_source}/commits/{commit_sha}")["tree"]["sha"]

    # get recursive tree
    tree_data = github_api_request(f"{data_source}/trees/{tree_sha}?recursive=1")[
        "tree"
    ]

    def is_valid_path(obj):
        return (
            ".joblib" in obj["path"].split("/")[-1]
            and ekh_version in obj["path"].split("/")[0]
        )

    print(
        "Available precomputed networks are:\n",
        *[
            '`ekh.river_network.load("{0}", "{1}")`\n'.format(
                *obj["path"].split("/")[1:3]
            )
            for obj in tree_data
            if is_valid_path(obj)
        ],
    )
