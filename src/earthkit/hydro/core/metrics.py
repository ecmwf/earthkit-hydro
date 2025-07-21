import numpy as np


def metrics_func_finder(metric, xp):

    class SumBased:
        func = xp.scatter_add
        base_val = 0

    metrics_dict = {
        "sum": SumBased,
    }
    return metrics_dict[metric]


# TODO: remove all this


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


metrics_dict = {
    "sum": SumBased,
    "mean": SumBased,
    "max": MaxBased,
    "min": MinBased,
    "prod": ProductBased,
    "std": SumBased,
    "var": SumBased,
}
