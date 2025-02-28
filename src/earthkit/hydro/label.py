import numpy as np

from .metrics import metrics_dict
from .utils import check_missing, is_missing


def calculate_metric_for_labels(
    field,
    labels,
    metric,
    field_mv=np.nan,
    labels_mv=0,
    field_accept_missing=False,
    missing_values_present=None,
):

    if missing_values_present is None:
        missing_values_present = check_missing(field, field_mv, field_accept_missing)

    if missing_values_present and not np.isnan(field_mv):
        # TODO: handle missing values
        raise NotImplementedError(
            "Support for generic missing values not yet implemented."
        )

    labels_ndim = labels.ndim
    fields_ndim = field.ndim  # fields_ndim = (extra_dims, *labels_ndim)

    labels = labels.T

    mask = ~is_missing(labels, labels_mv)
    not_missing_labels = labels[mask]

    expanded_mask = mask
    for _ in range(fields_ndim - labels_ndim):
        expanded_mask = expanded_mask[..., np.newaxis]

    relevant_field = field.T[mask]

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
        relevant_field,
    )

    if metric == "mean":
        count_values = np.bincount(
            unique_label_positions, minlength=len(unique_labels)
        ).astype(initial_field.dtype)
        initial_field_T = initial_field.T
        initial_field_T /= count_values
        initial_field = initial_field_T.T

    initial_field = np.transpose(
        initial_field, axes=[0] + list(range(initial_field.ndim - 1, 0, -1))
    )

    return dict(zip(unique_labels, initial_field))
