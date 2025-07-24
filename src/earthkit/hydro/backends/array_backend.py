class ArrayBackend:
    def __init__(self, module):
        self._mod = module

    def __getattr__(self, name):
        return getattr(self._mod, name)  # Delegate to underlying module

    @property
    def name(self):
        return self._mod.__name__.split(".")[0]

    def copy(self, x):
        raise NotImplementedError

    # extended functionality
    def scatter_assign(self, target, indices, updates):
        raise NotImplementedError

    def scatter_add(self, target, indices, updates):
        raise NotImplementedError

    def scatter_max(self, target, indices, updates):
        raise NotImplementedError

    def scatter_min(self, target, indices, updates):
        raise NotImplementedError

    def scatter_mul(self, target, indices, updates):
        raise NotImplementedError
