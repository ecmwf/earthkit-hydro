import numpy as np


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
    if np.isnan(mv):
        return np.isnan(field)
    elif np.isinf(mv):
        return np.isinf(field)
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
    return np.any(is_missing(field, mv))


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


def mask_2d(func):
    """Decorator to allow function to mask 2d inputs to the river network.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.

    """

    def wrapper(river_network, *args, **kwargs):
        """Wrapper masking 2d data fields to allow for processing along the
        river network, then undoing the masking.

        Parameters
        ----------
        river_network : object
            The RiverNetwork instance calling the method.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.

        """
        args = tuple(
            (
                arg[..., river_network.mask].T
                if isinstance(arg, np.ndarray)
                and arg.shape[-2:] == river_network.mask.shape
                else arg.T if isinstance(arg, np.ndarray) else arg
            )
            for arg in args
        )

        kwargs = {
            key: (
                value[..., river_network.mask].T
                if isinstance(value, np.ndarray)
                and value.shape[-2:] == river_network.mask.shape
                else value.T if isinstance(value, np.ndarray) else value
            )
            for key, value in kwargs.items()
        }

        return func(river_network, *args, **kwargs)

    return wrapper


def mask_and_unmask_data(func):
    """Decorator to convert masked 2d inputs back to 1d.

    Parameters
    ----------
    func : callable
        The function to be wrapped and executed with masking applied.

    Returns
    -------
    callable
        The wrapped function.

    """

    def wrapper(river_network, field, *args, **kwargs):
        """Wrapper masking 2d data fields to allow for processing along the
        river network, then undoing the masking.

        Parameters
        ----------
        river_network : object
            The RiverNetwork instance calling the method.
        field : numpy.ndarray
            The input data field to be processed.
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        numpy.ndarray
            The processed field.

        """
        # gets the missing value from the keyword arguments if it is present,
        # otherwise takes default value of mv from func
        mv = kwargs.get("mv")
        mv = mv if mv is not None else func.__defaults__[0]
        if field.shape[-2:] == river_network.mask.shape:
            in_place = kwargs.get("in_place", False)
            if in_place:
                out_field = field
            else:
                out_field = np.empty(field.shape, dtype=field.dtype)
            out_field[..., river_network.mask] = mask_2d(func)(
                river_network, field, *args, **kwargs
            ).T

            out_field[..., ~river_network.mask] = mv
            return out_field
        else:
            return mask_2d(func)(river_network, field, *args, **kwargs).T

    return wrapper
