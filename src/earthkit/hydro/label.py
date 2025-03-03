import numpy as np

from .metrics import metrics_dict
from .utils import check_missing, is_missing


def calculate_metric_for_labels(
    field,
    labels,
    metric,
    weights=None,
    field_mv=np.nan,
    labels_mv=0,
    field_accept_missing=False,
    missing_values_present_field=None,
    missing_values_present_weights=None,
):
    ufunc = metrics_dict[metric].func

    labels = labels.T

    mask = ~is_missing(labels, labels_mv)
    not_missing_labels = labels[mask]

    relevant_field = field[..., mask].T

    if missing_values_present_field is None:
        missing_values_present_field = check_missing(field, field_accept_missing)

    if weights is None:
        missing_values_present_weights = False
    else:
        relevant_weights = weights[..., mask].T
        relevant_field = (relevant_field.T * relevant_weights.T).T

    if missing_values_present_field is None:
        missing_values_present = check_missing(field, field_mv)
    else:
        missing_values_present = missing_values_present_field

    if missing_values_present_weights is None:
        missing_values_present_weights = check_missing(weights, field_mv)
        missing_values_present = (
            missing_values_present or missing_values_present_weights
        )
    else:
        missing_values_present = (
            missing_values_present or missing_values_present_weights
        )

    if missing_values_present and not np.isnan(field_mv):
        # TODO: handle missing values
        raise NotImplementedError(
            "Support for generic missing values is not yet implemented."
        )

    unique_labels, unique_label_positions = np.unique(
        not_missing_labels, return_inverse=True
    )

    initial_field = np.full(
        (len(unique_labels), *field.T.shape[labels.ndim :]),
        metrics_dict[metric].base_val,
        dtype=field.dtype,
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
                dtype=weights.dtype,
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

    return dict(zip(unique_labels, initial_field))
