import tempfile
import shutil
import os


class CacheManager:
    # Class-level variable for the cache directory
    cache_dir = tempfile.mkdtemp()

    def __init__(self, fn):
        self.name = fn.__name__
        self.fn = fn
        # we need to put the counter in a dict
        # or we lose the reference
        if self.name not in self.found:
            self.found[self.name] = 0

    def __call__(self, *args, **kwargs):
        key = "{}, {}, {}".format(self.name, args, kwargs)

        if key not in self.cache:
            data = self.fn(*args, **kwargs)
            # we don't cache small objects (e.g. floats from loadmap)
            if isinstance(data, float):
                return_data = data
            else:
                self.cache[key] = data
                return_data = self.cache[key]
        else:
            return_data = self.cache[key]
            self.found[self.name] += 1
        return return_data

    @classmethod
    def set_cache_dir(cls, path):
        # Set a new cache directory
        if os.path.exists(path):
            cls.cache_dir = path
        else:
            raise ValueError("The provided path does not exist")

    @classmethod
    def add_file(cls, filename, content):
        # Add a file to the cache directory
        file_path = os.path.join(cls.cache_dir, filename)
        with open(file_path, "w") as file:
            file.write(content)
        return file_path

    @classmethod
    def get_file(cls, filename):
        # Retrieve a file from the cache directory
        file_path = os.path.join(cls.cache_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return file.read()
        return None

    @classmethod
    def clear_cache(cls):
        # Delete all files in the cache directory
        shutil.rmtree(cls.cache_dir)
        # Recreate the temporary directory for the next session
        cls.cache_dir = tempfile.mkdtemp()
