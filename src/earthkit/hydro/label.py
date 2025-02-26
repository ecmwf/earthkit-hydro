import numpy as np

from .metrics import metrics
from .utils import check_missing, mask_and_unmask_data


@mask_and_unmask_data
def calculate_metric_for_labels(field, labels, metric, mv=np.nan, accept_missing=False):
    missing_values_present = check_missing(field, mv, accept_missing)
    if not missing_values_present or np.isnan(mv):
        pass
    else:
        # TODO: handle missing values
        raise NotImplementedError(
            "Support for generic missing values not yet implemented."
        )

    input_array = field.ravel()
    label_array = labels.ravel()

    unique_labels, unique_label_positions = np.unique(label_array, return_inverse=True)

    initial_field = np.full(len(unique_labels), metric.base_val, dtype=field.dtype)

    metrics.func.at(initial_field, unique_label_positions, input_array)

    if metric is metrics.mean:
        count_values = np.bincount(
            unique_label_positions, minlength=len(unique_labels)
        ).astype(initial_field.dtype)
        initial_field /= count_values

    return dict(zip(unique_labels, initial_field))
