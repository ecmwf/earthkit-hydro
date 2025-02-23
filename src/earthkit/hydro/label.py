import numpy as np

from .utils import check_missing


def calculate_metric_for_labels(
    field, labels, mv=np.nan, ufunc=np.add, accept_missing=False
):
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

    initial_field = np.full(
        len(unique_labels), get_identity_value(ufunc), dtype=field.dtype
    )

    if ufunc is np.mean:
        np.add.at(initial_field, unique_label_positions, input_array)
        count_values = np.bincount(
            unique_label_positions, minlength=len(unique_labels)
        ).astype(initial_field.dtype)
        initial_field /= count_values
    else:
        ufunc.at(initial_field, unique_label_positions, input_array)

    return dict(zip(unique_labels, initial_field))


def get_identity_value(ufunc):
    if ufunc is np.add or ufunc is np.mean or ufunc is np.subtract:
        return 0
    elif ufunc is np.multiply:
        return 1
    elif ufunc is np.maximum:
        return -np.inf
    elif ufunc is np.minimum:
        return np.inf
    else:
        raise NotImplementedError(f"ufunc {ufunc} is not implemented.")
