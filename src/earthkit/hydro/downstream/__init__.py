# SPDX-FileCopyrightText: 2026- European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

from earthkit.hydro.downstream import array

from ._toplevel import max, mean, min, mode, percentile, skewness, std, sum, var

__all__ = ["array", "max", "mean", "min", "mode", "percentile", "skewness", "std", "sum", "var"]
