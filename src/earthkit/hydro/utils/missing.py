import math

from earthkit.utils.array import array_namespace


def is_missing(field, mv):
    """Finds a mask of missing values.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.

    Returns
    -------
    numpy.ndarray
        A boolean mask of missing values.

    """
    xp = array_namespace(field)

    if math.isnan(mv):
        return xp.isnan(field)
    elif math.isinf(mv):
        return xp.isinf(field)
    else:
        return field == mv


def are_missing_values_present(field, mv):
    """Finds if missing values are present in a field.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.

    Returns
    -------
    bool
        True if missing values are present, False otherwise.

    """
    xp = array_namespace(field)
    return xp.any(is_missing(field, mv))


def check_missing(field, mv, accept_missing):
    """Finds missing values and checks if they are allowed in the input field.

    Parameters
    ----------
    field : numpy.ndarray
        The scalar input field to check for missing values.
    mv : scalar
        The missing value to check for.
    accept_missing : bool
        If True, missing values are allowed in the input field.

    Returns
    -------
    bool
        True if missing values are present, False otherwise.

    """
    missing_values_present = are_missing_values_present(field, mv)
    if missing_values_present:
        if not accept_missing:
            raise ValueError(
                "Missing values present in input field and accept_missing is False."
            )
        else:
            print("Warning: missing values present in input field.")
    return missing_values_present


def missing_to_nan(field, mv, accept_missing, skip=False):
    """
    Converts a field with arbitrary missing values to a field of type float with nans.

    Parameters
    ----------
    field : numpy.ndarray
       The input field.
    mv : scalar
        Missing values for the input field.
    accept_missing : bool
        If True, missing values are allowed in the input field.
    skip : bool, optional
        Skip this function. Default is False.

    Returns
    -------
    numpy.ndarray
        Output field.
    numpy.dtype
        dtype of the original field.


    """

    xp = array_namespace(field)

    if skip:
        return field, field.dtype

    missing_mask = is_missing(field, mv)

    if xp.any(missing_mask):
        if not accept_missing:
            raise ValueError(
                "Missing values present in input field and accept_missing is False."
            )
        else:
            print("Warning: missing values present in input field.")

    field_dtype = field.dtype
    if not field_dtype == xp.float64:
        field = field.astype(xp.float64, copy=False)  # convert to float64
    if xp.isnan(mv):
        return field, field_dtype
    field[missing_mask] = xp.nan
    return field, field_dtype


def nan_to_missing(out_field, field_dtype, mv):
    """
    Converts a floating field with np.nans back to original field
    with original missing values.

    Parameters
    ----------
    out_field : numpy.ndarray
       Field of type float with np.nans.
    field_dtype : numpy.dtype
        dtype to convert to.
    mv : scalar
        Original missing values.

    Returns
    -------
    numpy.ndarray
        Output field.

    """
    xp = array_namespace(out_field)
    if not xp.isnan(mv):
        xp.nan_to_num(out_field, copy=False, nan=mv)
    if field_dtype != xp.float64:
        out_field = out_field.astype(field_dtype, copy=False)
    return out_field
