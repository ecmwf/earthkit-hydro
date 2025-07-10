import numpy as np


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
                arg[..., river_network.mask]
                if isinstance(arg, np.ndarray) and arg.shape[-2:] == river_network.shape
                else arg if isinstance(arg, np.ndarray) else arg
            )
            for arg in args
        )

        kwargs = {
            key: (
                value[..., river_network.mask]
                if isinstance(value, np.ndarray)
                and value.shape[-2:] == river_network.shape
                else value if isinstance(value, np.ndarray) else value
            )
            for key, value in kwargs.items()
        }

        return func(river_network, *args, **kwargs)

    return wrapper


def mask_and_unmask(func):
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

        # skip! don't bother masking and unmasking
        # (if it has already been done)
        skip = kwargs.pop("skip", None)
        skip = skip if skip is not None else False
        if skip:
            return func(river_network, field, *args, **kwargs)

        # gets the missing value from the keyword arguments if it is present,
        # otherwise takes default value of mv from func
        mv = kwargs.get("mv")
        mv = mv if mv is not None else func.__defaults__[0]
        if field.shape[-2:] == river_network.shape:
            in_place = kwargs.get("in_place", False)

            values_on_river_network = mask_2d(func)(
                river_network, field, *args, **kwargs
            )

            if in_place:
                out_field = field
            else:
                out_field = np.empty(field.shape, dtype=values_on_river_network.dtype)

            out_field[..., river_network.mask] = values_on_river_network

            if np.result_type(mv, field) != field.dtype:
                raise ValueError(
                    f"Missing value of type {type(mv)} is not compatible"
                    f" with field of dtype {field.dtype}"
                )

            out_field[..., ~river_network.mask] = mv
            return out_field
        else:
            return mask_2d(func)(river_network, field, *args, **kwargs)

    return wrapper
