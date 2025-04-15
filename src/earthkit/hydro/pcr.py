from struct import unpack

import numpy as np

# CSF value scales
# version 2 datatypes
VS_BOOLEAN = 0xE0  # boolean, always UINT1, values: 0,1 or MV_UINT1
VS_NOMINAL = 0xE2  # nominal, UINT1 or INT4
VS_ORDINAL = 0xF2  # ordinal, UINT1 or INT4
VS_SCALAR = 0xEB  # scalar, REAL4 or (maybe) REAL8
VS_DIRECTION = 0xFB  # directional REAL4 or (maybe) REAL8, -1 means no direction
VS_LDD = 0xF0  # local drain direction, always UINT1, values: 1-9 or MV_UINT1
# this one CANNOT be returned by NOR passed to a csf2 function
VS_UNDEFINED = 100  # just some value different from the rest

# CSF cell representations
# preferred version 2 cell representations
CR_UINT1 = 0x00  # boolean, ldd and small nominal and small ordinal
CR_INT4 = 0x26  # large nominal and large ordinal
CR_REAL4 = 0x5A  # single scalar and single directional
# other version 2 cell representations
CR_REAL8 = 0xDB  # double scalar or directional, no loss of precision


def _replace_missing_u1(cur, new):
    out = np.copy(cur)
    out[cur == 255] = new
    return out


def _replace_missing_i4(cur, new):
    out = np.copy(cur)
    out[cur == -2147483648] = new
    return out


def _replace_missing_f4(cur, new):
    out = np.copy(cur)
    out[np.isnan(cur)] = new
    return out


def _replace_missing_f8(cur, new):
    out = np.copy(cur)
    out[np.isnan(cur)] = new
    return out


CELLREPR = {
    CR_UINT1: dict(
        dtype=np.dtype("uint8"),
        fillmv=_replace_missing_u1,
    ),
    CR_INT4: dict(
        dtype=np.dtype("int32"),
        fillmv=_replace_missing_i4,
    ),
    CR_REAL4: dict(
        dtype=np.dtype("float32"),
        fillmv=_replace_missing_f4,
    ),
    CR_REAL8: dict(
        dtype=np.dtype("float64"),
        fillmv=_replace_missing_f8,
    ),
}


def from_file(path, mask=False):
    """
    Load a .map file from numpy array.
    """

    with open(path, "rb") as f:
        bytes = f.read()

    nbytes_header = 64 + 2 + 2 + 8 + 8 + 8 + 8 + 4 + 4 + 8 + 8 + 8
    _, cellRepr, _, _, _, _, nrRows, nrCols, _, _, _ = unpack(
        "=hhddddIIddd", bytes[64:nbytes_header]
    )

    try:
        celltype = CELLREPR[cellRepr]
    except KeyError:
        raise Exception(
            "{}: invalid cellRepr value ({}) in header".format(path, cellRepr)
        )

    dtype = celltype["dtype"]

    size = dtype.itemsize * nrRows * nrCols
    data = np.frombuffer(bytes[256 : 256 + size], dtype)
    if mask:
        data = celltype["fillmv"](data, np.nan)

    return data
