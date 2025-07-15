from earthkit.hydro.core import propagate
from earthkit.hydro.core._find import _find_catchments_2D
from earthkit.hydro.utils.decorators import xarray_mask_and_unmask


@xarray_mask_and_unmask
def find(river_network, field, mv=0, in_place=False):
    """Labels the catchments given a field of labelled sinks.

    Parameters
    ----------
    river_network : earthkit.hydro.network.RiverNetwork
        An earthkit-hydro river network object.
    field : numpy.ndarray
        The input field.
    mv : scalar, optional
        The missing value indicator. Default is 0.
    in_place : bool, optional
        If True, modifies the input field in place. Default is False.

    Returns
    -------
    numpy.ndarray
        The field values accumulated downstream.

    """
    if not in_place:
        field = field.copy()

    if len(field.shape) == 1:
        op = _find_catchments_2D
    else:
        raise NotImplementedError("ND arrays not yet supported.")

    def operation(field, did, uid, eid, mv):
        return op(field, did, uid, eid, mv, overwrite=True)

    return propagate(river_network, field, True, operation, mv)
