import shutil
from urllib.request import urlretrieve
import os


class Cache:
    def __init__(self, cache_dir, data_source, cache_fname):
        self.cache_dir = cache_dir
        self.data_source = data_source
        self.cache_fname = cache_fname
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        print("caching in", self.cache_dir)

    def __call__(self, **kwargs):
        filepath = os.path.join(self.cache_dir.format(**kwargs), self.cache_fname.format(**kwargs))
        sourcepath = self.data_source.format(**kwargs)
        if not os.path.isfile(filepath):
            if self.data_source.startswith("http"):
                urlretrieve(sourcepath, filepath)
            else:
                shutil.copy(sourcepath, filepath)
        return filepath

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)
        print("cache cleared")

    def delete_cache(self):
        shutil.rmtree(self.cache_dir)
        print("cache deleted")
