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

    missing_values_present_weights = (
        False if weights is None else missing_values_present_weights
    )

    if missing_values_present_field is None:
        missing_values_present = check_missing(field, field_mv, field_accept_missing)
    if missing_values_present_weights is None:
        missing_values_present = (
            check_missing(weights, field_mv, field_accept_missing)
            or missing_values_present
        )

    if missing_values_present and not np.isnan(field_mv):
        # TODO: handle missing values
        raise NotImplementedError(
            "Neither support for weights nor generic missing values is implemented yet."
        )

    labels = labels.T

    mask = ~is_missing(labels, labels_mv)
    not_missing_labels = labels[mask]

    relevant_field = field[..., mask].T

    if weights is not None:
        relevant_weights = weights[..., mask].T

    unique_labels, unique_label_positions = np.unique(
        not_missing_labels, return_inverse=True
    )

    initial_field = np.full(
        (len(unique_labels), *field.T.shape[labels.ndim :]),
        metrics_dict[metric].base_val,
        dtype=field.dtype,
    )

    metrics_dict[metric].func.at(
        initial_field,
        (unique_label_positions, *[slice(None)] * (initial_field.ndim - 1)),
        (
            relevant_field
            if weights is None
            else (relevant_field.T * relevant_weights.T).T
        ),
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
            metrics_dict[metric].func.at(
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
