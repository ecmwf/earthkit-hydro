import os
import tempfile
from functools import wraps
from hashlib import sha256

import joblib

from earthkit.hydro._version import __version__ as ekh_version
from earthkit.hydro.data_structures._network import RiverNetwork

# read in only up to second decimal point
# i.e. 0.1.dev90 -> 0.1
ekh_version = ".".join(ekh_version.split(".")[:2])


def cache(func):
    """
    Decorator to allow automatic use of cache.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.
    """

    @wraps(func)
    def wrapper(
        path,
        river_network_format,
        source="file",
        use_cache=True,
        cache_dir=tempfile.mkdtemp(suffix="_earthkit_hydro"),
        cache_fname="{ekh_version}_{hash}.joblib",
        cache_compression=1,
    ):
        """
        Wrapper to load river network from cache if available, otherwise
        create and cache it.

        Parameters
        ----------
        path : str
            The path to the river network.
        river_network_format : str
            The format of the river network file.
            Supported formats are "precomputed", "cama", "pcr_d8", and "esri_d8".
        source : str, optional
            The source of the river network.
            For possible sources see:
            https://earthkit-data.readthedocs.io/en/latest/guide/sources.html
        use_cache : bool, optional
            Whether to use caching. Default is True.
        cache_dir : str, optional
            The directory to store the cache files. Default is a temporary directory.
        cache_fname : str, optional
            The filename template for the cache files.
            Default is "{ekh_version}_{hash}.joblib".
        cache_compression : int, optional
            The compression level for the cache files. Default is 1.

        Returns
        -------
        earthkit.hydro.network_class.RiverNetwork
            The loaded river network.
        """
        if use_cache:
            hashed_name = sha256(path.encode("utf-8")).hexdigest()
            cache_dir = cache_dir.format(ekh_version=ekh_version, hash=hashed_name)
            cache_fname = cache_fname.format(ekh_version=ekh_version, hash=hashed_name)
            cache_filepath = os.path.join(cache_dir, cache_fname)

            if os.path.isfile(cache_filepath):
                print(f"Loading river network from cache ({cache_filepath}).")
                return RiverNetwork(joblib.load(cache_filepath))
            else:
                print(f"River network not found in cache ({cache_filepath}).")
                os.makedirs(cache_dir, exist_ok=True)
        else:
            print("Cache disabled.")

        network = func(path, river_network_format, source)

        if use_cache:
            joblib.dump(network._storage, cache_filepath, compress=cache_compression)
            print(f"River network loaded, saving to cache ({cache_filepath}).")

        return network

    return wrapper
