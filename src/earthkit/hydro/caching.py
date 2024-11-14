import shutil
from urllib.request import urlretrieve
import os


class Cache:
    """
    A class for caching data files locally from a specified source.

    Attributes
    ----------
    cache_dir : str
        Directory where cached files are stored.
    data_source : str
        Source URL or file path template for retrieving data.
    cache_fname : str
        Template for the cache file name.

    Methods
    -------
    __call__(**kwargs)
        Retrieves a file from the source and caches it locally if it doesn't already exist.
    clear_cache()
        Clears all files in the cache directory.
    delete_cache()
        Deletes the entire cache directory.
    """

    def __init__(self, cache_dir, data_source, cache_fname):
        """
        Initialises the Cache object with a specified cache directory, data source, and cache file name template.

        Parameters
        ----------
        cache_dir : str
            Directory where cached files will be stored.
        data_source : str
            Template for the source URL or file path to retrieve data.
        cache_fname : str
            Template for the cache file name.
        """
        self.cache_dir = cache_dir
        self.data_source = data_source
        self.cache_fname = cache_fname
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        print("caching in", self.cache_dir)

    def __call__(self, **kwargs):
        """
        Retrieves a file from the data source and caches it locally.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to format the data source and cache file name templates.

        Returns
        -------
        str
            The file path of the cached file.

        Notes
        -----
        If the data source is an HTTP URL, the file is downloaded.
        If the data source is a local file, it is copied.
        """
        filepath = os.path.join(self.cache_dir.format(**kwargs), self.cache_fname.format(**kwargs))
        sourcepath = self.data_source.format(**kwargs)
        if not os.path.isfile(filepath):
            if self.data_source.startswith("http"):
                urlretrieve(sourcepath, filepath)
            else:
                shutil.copy(sourcepath, filepath)
        return filepath

    def clear_cache(self):
        """
        Clears all files in the cache directory.
        """
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
        print("cache cleared")

    def delete_cache(self):
        """
        Deletes the entire cache directory.
        """
        shutil.rmtree(self.cache_dir)
        print("cache deleted")
