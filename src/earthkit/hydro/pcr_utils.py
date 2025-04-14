import numpy as np
import logging
from struct import unpack, pack

# ----------------------------------------
# functions to read/write numpy objects
# ----------------------------------------

log = logging.getLogger(__name__)


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


def toPCRMap(func, template):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            result = PCRMap.from_template(result, template)
        except ValueError:
            pass
        return result
    return wrapper


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


class PCRMap(object):
    """
    Class to read/write pcraster maps
    The object contains the map data and attributes
    """

    def __init__(
        self,
        signature,
        valueScale,
        cellRepr,
        minVal,
        maxVal,
        xUL,
        yUL,
        nrRows,
        nrCols,
        cellSizeX,
        cellSizeY,
        angle,
        predata,
        data,
    ):
        """Create PCRaster object from header and data."""

        self._meta = {
            "signature": signature,
            "valueScale": valueScale,
            "cellRepr": cellRepr,
            "minVal": minVal,
            "maxVal": maxVal,
            "xUL": xUL,
            "yUL": yUL,
            "nrRows": nrRows,
            "nrCols": nrCols,
            "cellSizeX": cellSizeX,
            "cellSizeY": cellSizeY,
            "angle": angle,
            "predata": predata,
        }
        self.data = data.reshape(nrRows, nrCols)

        self.dtype = CELLREPR[self._meta["cellRepr"]]["dtype"]

    def fillMV(self, filler):
        """Return NumPy array with raster missing values replaced by 'filler'."""
        return CELLREPR[self.self._meta["cellRepr"]]["fillmv"](self.data, filler)

    @classmethod
    def from_file(cls, path):
        """
        Returns a PCRMap object from PCRaster map file.
        """

        log.debug("Reading PCRaster file " + path)

        with open(path, "rb") as f:
            bytes = f.read()

        # read raster header
        signature = bytes[:64]
        nbytes_header = 64 + 2 + 2 + 8 + 8 + 8 + 8 + 4 + 4 + 8 + 8 + 8
        valueScale, cellRepr, minVal, maxVal, xUL, yUL, nrRows, nrCols, cellSizeX, cellSizeY, angle = unpack(
            "=hhddddIIddd", bytes[64:nbytes_header]
        )
        predata = bytes[nbytes_header:256]

        # read data
        try:
            dtype = CELLREPR[cellRepr]["dtype"]
        except KeyError:
            raise Exception("{}: invalid cellRepr value ({}) in header".format(path, cellRepr))
        size = dtype.itemsize * nrRows * nrCols
        data = np.frombuffer(bytes[256 : 256 + size], dtype)

        return PCRMap(
            signature,
            valueScale,
            cellRepr,
            minVal,
            maxVal,
            xUL,
            yUL,
            nrRows,
            nrCols,
            cellSizeX,
            cellSizeY,
            angle,
            predata,
            data,
        )

    @classmethod
    def from_template(cls, data, template):
        """
        Returns a PCRMap object from data and template.
        """

        return PCRMap(
            **template._meta,
            data=data,
        )

    def write(self, path):

        with open(path, "wb") as f:
            f.write(self.signature)

            bytes_metadata = pack(
                "=hhddddIIddd",
                self._meta["valueScale"],
                self._meta["cellRepr"],
                self._meta["minVal"],
                self._meta["maxVal"],
                self._meta["xUL"],
                self._meta["yUL"],
                self._meta["nrRows"],
                self._meta["nrCols"],
                self._meta["cellSizeX"],
                self._meta["cellSizeY"],
                self._meta["angle"],
            )
            f.write(bytes_metadata)

            f.write(self._meta["predata"])

            bytes_data = self.data.ravel().tobytes()
            f.write(bytes_data)

    def __repr__(self):
        return f"PCRMap({self._meta})\n{self.data}"

    def __getitem__(self, item):
        result = self.data[item]
        if isinstance(result, np.ndarray):
            result = PCRMap.from_template(result, self)
        return result

    def __getattr__(self, item):
        if item == "_meta":
            return self._meta
        result = getattr(self.data, item)
        if callable(result):
            result = toPCRMap(result, self)
        return result

    def __add__(self, other):
        return PCRMap.from_template(self.data + other, self)
    def __sub__(self, other):
        return PCRMap.from_template(self.data - other, self)
    def __truediv__(self, other):
        return PCRMap.from_template(self.data / other, self)
    def __mul__(self, other):
        return PCRMap.from_template(self.data * other, self)
    def __mod__(self, other):
        return PCRMap.from_template(self.data % other, self)
    def __pow__(self, other):
        return PCRMap.from_template(self.data ** other, self)
    def __floordiv__(self, other):
        return PCRMap.from_template(self.data // other, self)
    def __matmul__(self, other):
        return PCRMap.from_template(self.data @ other, self)
    def __radd__(self, other):
        return PCRMap.from_template(self.data + other, self)
    def __rsub__(self, other):
        return PCRMap.from_template(other - self.data, self)
    def __rtruediv__(self, other):
        return PCRMap.from_template(other / self.data, self)
    def __rmul__(self, other):
        return PCRMap.from_template(self.data * other, self)
    def __rmod__(self, other):
        return PCRMap.from_template(other % self.data, self)
    def __rpow__(self, other):
        return PCRMap.from_template(other ** self.data, self)
    def __rfloordiv__(self, other):
        return PCRMap.from_template(other // self.data, self)
    def __rmatmul__(self, other):
        return PCRMap.from_template(other @ self.data, self)
    def __and__(self, other):
        return PCRMap.from_template(self.data & other, self)
    def __or__(self, other):
        return PCRMap.from_template(self.data | other, self)
    def __xor__(self, other):
        return PCRMap.from_template(other ^ self.data, self)
    def __invert__(self):
        return PCRMap.from_template(~self.data, self)
    def __neg__(self):
        return PCRMap.from_template(-self.data, self)
    def __pos__(self):
        return PCRMap.from_template(+self.data, self)
    def __abs__(self):
        return PCRMap.from_template(abs(self.data), self)
    