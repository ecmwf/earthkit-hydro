import numpy as np


class Metric:
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"Metric({self.name}, {self.func})"


class Metrics:
    sum = Metric("sum", np.add)
    mean = Metric("mean", np.add)
    max = Metric("max", np.maximum)
    min = Metric("min", np.minimum)

    def __getattr__(self, name):
        # Check if the requested attribute exists in the class
        if name not in self.__class__.__dict__:
            raise NotImplementedError(
                f"Invalid metric '{name}'. Available metrics are: sum, mean, max, min"
            )
        return getattr(self, name)


metrics = Metrics()
