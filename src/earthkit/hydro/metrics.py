import numpy as np


class Sum:
    func = np.add


class Mean:
    func = np.add


class Max:
    func = np.maximum


class Min:
    func = np.minimum


metrics_dict = {"sum": Sum, "mean": Mean, "max": Max, "min": Min}
