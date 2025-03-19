from functools import partial

import numpy as np

from .accumulation import flow_downstream
from .metrics import metrics_dict
from .utils import mask_and_unmask, missing_to_nan, nan_to_missing


@mask_and_unmask
def calculate_upstream_metric(
    river_network,
    field,
    metric,
    weights=None,
    mv=np.nan,
    in_place=False,
    accept_missing=False,
):
    """
    Calculates a metric for the field over all upstream values.

    Parameters
    ----------
    river_network : earthkit.hydro.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    metric : str
        Metric to compute. Options are "mean", "max", "min", "sum", "product"
    weights : ndarray, optional
        Used to weight the field when computing the metric. Default is None.
    mv : scalar, optional
        Missing value for the input field. Default is np.nan.
    in_place : bool, optional
        Whether to conduct the operation in-place. Default is False.
    accept_missing : bool, optional
        Whether or not to accept missing values in the input field. Default is False.

    Returns
    -------
    numpy.ndarray
        Output field.

    """
    field, field_dtype = missing_to_nan(field, mv, accept_missing)
    if weights is None:
        weights = np.ones(river_network.n_nodes, dtype=np.float64)
    else:
        assert field_dtype == weights.dtype
        weights, _ = missing_to_nan(weights, mv, accept_missing)
        field = (field.T * weights.T).T

    ufunc = metrics_dict[metric].func

    field = flow_downstream(
        river_network,
        field,
        np.nan,  # mv replaced by nan
        in_place,
        ufunc,
        accept_missing,
        skip_missing_check=True,
        skip=True,
    )

    if metric == "mean":
        counts = flow_downstream(
            river_network,
            weights,
            np.nan,  # mv replaced by nan
            in_place,
            ufunc,
            accept_missing,
            skip_missing_check=True,
            skip=True,
        )
        field_T = field.T
        field_T /= counts.T
        return nan_to_missing(
            field_T.T, np.float64, mv
        )  # if we compute means, we change dtype for int fields etc.
    else:
        return nan_to_missing(field, field_dtype, mv)


for metric in metrics_dict.keys():

    func = partial(calculate_upstream_metric, metric=metric)

    globals()[metric] = func
