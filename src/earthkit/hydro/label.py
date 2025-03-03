import numpy as np

from .metrics import metrics_dict
from .utils import is_missing, missing_to_nan, nan_to_missing


def calculate_metric_for_labels(
    field,
    labels,
    metric,
    weights=None,
    field_mv=np.nan,
    labels_mv=0,
    field_accept_missing=False,
    skip_missing_check=False,
):
    ufunc = metrics_dict[metric].func

    mask = ~is_missing(labels, labels_mv)
    not_missing_labels = labels[mask].T

    relevant_field = field[..., mask].T

    relevant_field, field_dtype = missing_to_nan(
        relevant_field, field_mv, field_accept_missing, skip_missing_check
    )

    if weights is not None:
        assert field_dtype == weights.dtype
        relevant_weights = weights[..., mask].T
        relevant_weights, _ = missing_to_nan(
            relevant_weights, field_mv, field_accept_missing, skip_missing_check
        )
        relevant_field = (relevant_field.T * relevant_weights.T).T

    unique_labels, unique_label_positions = np.unique(
        not_missing_labels, return_inverse=True
    )

    initial_field = np.full(
        (len(unique_labels), *field.T.shape[labels.ndim :]),
        metrics_dict[metric].base_val,
        dtype=float,
    )

    ufunc.at(
        initial_field,
        (unique_label_positions, *[slice(None)] * (initial_field.ndim - 1)),
        relevant_field,
    )

    if metric == "mean":
        if weights is None:
            count_values = np.bincount(
                unique_label_positions, minlength=len(unique_labels)
            ).astype(initial_field.dtype)
        else:
            count_values = np.full(
                (len(unique_labels), *weights.T.shape[labels.ndim :]),
                metrics_dict[metric].base_val,
                dtype=relevant_weights.dtype,
            )
            ufunc.at(
                count_values,
                (unique_label_positions, *[slice(None)] * (count_values.ndim - 1)),
                relevant_weights,
            )
        initial_field_T = initial_field.T
        initial_field_T /= count_values.T
        initial_field = initial_field_T.T

    initial_field = np.transpose(
        initial_field, axes=[0] + list(range(initial_field.ndim - 1, 0, -1))
    )

    initial_field = nan_to_missing(initial_field, field_dtype, field_mv)

    return dict(zip(unique_labels, initial_field))
