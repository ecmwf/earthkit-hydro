import numpy as np


class SumBased:
    func = np.add
    base_val = 0


class MaxBased:
    func = np.maximum
    base_val = -np.inf


class MinBased:
    func = np.minimum
    base_val = np.inf


class ProductBased:
    func = np.multiply
    base_val = 1


class Percentile:
    func = np.percentile


metrics_dict = {
    "sum": SumBased,
    "mean": SumBased,
    "max": MaxBased,
    "min": MinBased,
    "prod": ProductBased,
    "std": SumBased,
    "var": SumBased,
    "percentile": Percentile,
}
