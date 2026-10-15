"""Access IMD (India Meteorological Department) gridded daily rainfall.

Source: IMD Pune 0.25deg x 0.25deg gauge-based daily rainfall (Pai et al. 2014),
downloaded directly from IMD via the ``imdlib`` package. This is the gauge-based
reference product for India -- coarser than GSMaP (~28 km vs ~11 km) but ground-
truth rather than satellite estimate.

Grid: lat 6.5-38.5N (129), lon 66.5-100.0E (135). Value = daily total rainfall in
mm (NO x24 conversion, unlike GSMaP). Nodata (ocean / outside India) = -999.
"""
from __future__ import annotations

import socket

import imdlib
import numpy as np
import pandas as pd

socket.setdefaulttimeout(60)   # bound hangs against the flaky IMD real-time server

FILL_BELOW = -900.0   # -999 nodata sentinel
RES = 0.25


def year_stack(year: int, bbox: tuple[float, float, float, float], cache_dir: str):
    """Whole-year IMD rainfall (mm/day) over ``bbox``: (stack[T,nlat,nlon], times, lats, lons).

    Nodata (-999) becomes NaN. Uses the local cache if present, otherwise downloads.
    """
    try:
        d = imdlib.open_data("rain", year, year, "yearwise", file_dir=cache_dir)
    except Exception:  # noqa: BLE001 -- not cached yet
        d = imdlib.get_data("rain", year, year, fn_format="yearwise", file_dir=cache_dir)
    da = d.get_xarray()["rain"]
    da = da.where(da > FILL_BELOW)
    minx, miny, maxx, maxy = bbox
    da = da.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))
    stack = np.asarray(da.values, dtype="float64")
    times = pd.to_datetime(da["time"].values)
    lats = np.asarray(da["lat"].values, dtype="float64")
    lons = np.asarray(da["lon"].values, dtype="float64")
    return stack, times, lats, lons
