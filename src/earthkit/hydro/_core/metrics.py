def metrics_func_finder(metric, xp):

    class SumBased:
        func = xp.scatter_add
        base_val = 0

    class MaxBased:
        func = xp.scatter_max
        base_val = -xp.inf

    class MinBased:
        func = xp.scatter_min
        base_val = xp.inf

    metrics_dict = {
        "sum": SumBased,
        "mean": SumBased,
        "std": SumBased,
        "var": SumBased,
        "max": MaxBased,
        "min": MinBased,
    }
    return metrics_dict[metric]
