import numpy as np


class Sum:
    func = np.add
    base_val = 0


class Mean:
    func = np.add
    base_val = 0


class Max:
    func = np.maximum
    base_val = -np.inf


class Min:
    func = np.minimum
    base_val = np.inf


class Product:
    func = np.multiply
    base_val = 1


metrics_dict = {"sum": Sum, "mean": Mean, "max": Max, "min": Min, "product": Product}
