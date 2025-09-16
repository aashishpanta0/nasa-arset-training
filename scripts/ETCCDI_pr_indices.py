"""
ETCCDI precipitation (pr) indices
---------------------------------
Subroutines to compute common ETCCDI precipitation indices using **daily precipitation**
(`pr`: xarray.DataArray with a "time" dimension). The style mirrors
`ETCCDI_tasmin_indices.py` and `ETCCDI_tx_indices.py` in this project.

References
----------
ETCCDI list: https://etccdi.pacificclimate.org/list_27_indices.shtml

Requirements
------------
- xarray
- numpy
- pandas
- dask (optional)

Input conventions
-----------------
- pr: xarray.DataArray (daily precipitation) with a "time" dimension.
- Units should be mm/day (or kg m-2 day-1). If units are in m/day, we convert to mm/day.
- Time coordinate should be daily and monotonic.

Implemented indices (precipitation)
-----------------------------------
- Rx1day: Period maximum 1-day precipitation amount (mm) — monthly/annual/seasonal
- Rx5day: Period maximum consecutive 5-day precipitation (mm)
- SDII:   Simple Daily Intensity Index — mean precipitation on wet days (mm/day)
- R10mm:  Number of heavy precipitation days (>= 10 mm)
- R20mm:  Number of very heavy precipitation days (>= 20 mm)
- CDD:    Consecutive Dry Days — max number of consecutive days with pr < 1 mm
- CWD:    Consecutive Wet Days — max number of consecutive days with pr >= 1 mm
- PRCPTOT: Total precipitation in wet days (>= 1 mm) (mm)
- R95pTOT: Total precipitation from very wet days (> 95th percentile of wet-day pr) (mm)
- R99pTOT: Total precipitation from extremely wet days (> 99th percentile of wet-day pr) (mm)

Percentile-based indices follow ETCCDI:
- Thresholds are computed on **wet days only** (pr >= wet_day_thresh, default 1 mm)
  for each calendar day using a 5-day moving window (±2 days) over a baseline period
  (default 1961–1990). Thresholds are then applied to the full analysis period.

"""


from __future__ import annotations

__DISCLAIMER__ = "ALL INDICES PRODUCED BY THESE SCRIPTS/MODULES ARE EXPERIMENTAL. USERS MUST REVIEW THE CODE AND VALIDATE RESULTS AGAINST KNOWN REFERENCES BEFORE ANY OPERATIONAL OR DECISION SUPPORT USE. THERE MAY BE ERRORS."
import numpy as np
import numba as nb
import pandas as pd
import xarray as xr

def _finalize_index_like(_src, _out, _name, _units):
    import xarray as _xr
    # Order dims
    try:
        order = tuple(d for d in ("time","lat","lon") if d in _out.dims)
        if order:
            _out = _out.transpose(*order)
    except Exception:
        pass
    # Align coords and exact grid to source
    try:
        if "lat" in _out.dims and "lat" in _src.dims:
            _out = _out.assign_coords(lat=_src["lat"]).reindex(lat=_src["lat"])
        if "lon" in _out.dims and "lon" in _src.dims:
            _out = _out.assign_coords(lon=_src["lon"]).reindex(lon=_src["lon"])
    except Exception:
        pass
    # Mask-all-NaN across time
    try:
        if "time" in _src.dims and "time" in _out.dims:
            _mask_obs = _src.notnull().any("time")
            _out = _out.where(_mask_obs)
    except Exception:
        pass
    # Enforce chunking: (time:1, lat:all, lon:all)
    try:
        chunks = {}
        if "time" in _out.dims: chunks["time"] = 1
        if "lat" in _out.dims: chunks["lat"] = -1
        if "lon" in _out.dims: chunks["lon"] = -1
        if chunks:
            _out = _out.chunk(chunks)
    except Exception:
        pass
    _out = _out.astype("float32")
    _out.name = _name
    _out.attrs["units"] = _units
    return _out
from typing import Tuple, Literal

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _to_mm_per_day(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure precipitation in mm/day.
    Accepts units in {"mm/day", "kg m-2 day-1", "kg/m^2/day", "mm d-1"} or meters/day ("m/day") -> mm/day.
    If units unknown, leave values as-is and set to 'mm/day'.
    """
    units = (str(da.attrs.get("units", "")).lower() or "")
    # Normalize some variations
    units = units.replace("kg m-2 day-1", "kg/m2/day").replace("kg m^-2 day^-1", "kg/m2/day")
    units = units.replace("mm d-1", "mm/day").replace("mm/day", "mm/day").replace("mm per day", "mm/day")
    units = units.replace("m d-1", "m/day").replace("m per day", "m/day")
    if "m/day" in units and "mm/day" not in units:
        out = da * 1000.0
        out.attrs["units"] = "mm/day"
        return out
    if ("mm/day" in units) or ("kg/m2/day" in units):
        da.attrs["units"] = "mm/day"
        return da
    if ("kg m-2 s-1" in units) or ("kg/m2/s" in units) or ("mm/s" in units):
        out = da * 86400.0
        out.attrs["units"] = "mm/day"
        return out
    # Unknown: assume already mm/day
    out = da.copy()
    out.attrs["units"] = "mm/day"
    return out


def _resample_reduce(da: xr.DataArray, freq: str, reduce: str) -> xr.DataArray:
    """Generic reducer for a given resampling frequency."""
    r = da if freq == "D" else da.resample(time=freq)
    if reduce == "max":
        return r.max(keep_attrs=True)
    if reduce == "min":
        return r.min(keep_attrs=True)
    if reduce == "sum":
        return r.sum(keep_attrs=True)
    if reduce == "mean":
        return r.mean(keep_attrs=True)
    raise ValueError(f"Unsupported reduce={reduce}")


def _coarsen_by_period(period: Literal["monthly","annual","seasonal"]) -> str:
    """Map simple period keywords to resample frequencies (DJF, MAM, JJA, SON for seasonal)."""
    if period == "monthly":
        return "MS"
    if period == "annual":
        return "YS"
    if period == "seasonal":
        return "QS-DEC"  # DJF, MAM, JJA, SON with labels at season start
    raise ValueError("period must be one of {'monthly','annual','seasonal'}")


def _rolling_sum_5day(pr: xr.DataArray) -> xr.DataArray:
    """5-day rolling sum aligned to the **end** of the window (like ETCCDI Rx5day)."""
    return pr.rolling(time=5, min_periods=5).sum()


def _wetday_mask(pr: xr.DataArray, wet_day_thresh: float = 1.0) -> xr.DataArray:
    """Boolean mask of wet days (pr >= wet_day_thresh mm/day)."""
    return pr >= wet_day_thresh


def _wet_percentile_thresholds_calendar_day(
    pr: xr.DataArray,
    base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
    q: float = 95.0,
    window: int = 5,
    wet_day_thresh: float = 1.0,
) -> xr.DataArray:
    """
    Compute calendar-day percentile thresholds on **wet days only** with a moving window
    over a baseline period. Returns thresholds indexed by day-of-year (1..365). Leap day -> 365.
    """
    if window % 2 != 1:
        raise ValueError("window must be an odd integer (e.g., 5, 7, 11)")
    pr = _to_mm_per_day(pr)
    base = pr.sel(time=slice(base_period[0], base_period[1]))
    if base.time.size == 0:
        raise ValueError("Baseline period selection is empty. Check base_period and data time range.")
    doy = base["time"].dt.dayofyear
    doy = xr.where(doy == 366, 365, doy)
    base = base.assign_coords(doy=("time", doy.values))
    # Only wet days
    base = base.where(base >= wet_day_thresh)

    half = window // 2
    doys = np.arange(1, 366)
    sample = base.isel(time=0).drop_vars([v for v in ["doy"] if v in base.coords])
    shp = (365,) + tuple(sample.shape)
    thresh = np.full(shp, np.nan, dtype=float)

    for i, d in enumerate(doys):
        lo, hi = d - half, d + half
        window_doys = ((np.arange(lo, hi + 1) - 1) % 365) + 1
        sel = base.where(base["doy"].isin(window_doys), drop=True)
        arr = sel.transpose(..., "time").values
        if np.all(np.isnan(arr)):
            continue
        thresh[i, ...] = np.nanpercentile(arr, q, axis=-1)

    out = xr.DataArray(
        thresh,
        dims=("doy",) + sample.dims,
        coords={"doy": doys, **{k: v for k in sample.coords.keys()}},
        attrs={
            "description": f"{q}th percentile threshold by calendar day (wet days only)",
            "units": "mm/day",
        },
    )
    return out


def _apply_doy_threshold(da: xr.DataArray, thresh_doy: xr.DataArray, op: str = ">") -> xr.DataArray:
    """Compare daily values to DOY-indexed thresholds and return a boolean mask."""
    doy = da["time"].dt.dayofyear
    doy = xr.where(doy == 366, 365, doy)
    t = thresh_doy.sel(doy=doy)
    if op == ">":
        return da > t
    if op == "<":
        return da < t
    if op == ">=":
        return da >= t
    if op == "<=":
        return da <= t
    raise ValueError(f"Unsupported op={op}")


@nb.njit
def _runlen_1d(x):
    if x.size == 0:
        return 0
    maxlen = 0
    cur = 0
    for i in range(x.size):
        if x[i] == 1:
            cur += 1
            if cur > maxlen:
                maxlen = cur
        else:
            cur = 0
    return maxlen


def _max_run_length(mask: xr.DataArray) -> xr.DataArray:
    """
    Return, for each 1D array, the **maximum run length** of consecutive True values.
    """
    return xr.apply_ufunc(
        _runlen_1d, mask,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
        dask_gufunc_kwargs=dict(allow_rechunk=True)
    )

# --------------------------------------------------------------------------------------


def Rx1day(pr: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Maximum 1-day precipitation per period (mm)."""
    pr = _to_mm_per_day(pr)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(pr, freq=freq, reduce="max")
    out.name = "Rx1day"
    out.attrs["long_name"] = f"{period.capitalize()} maximum 1-day precipitation"
    out.attrs["units"] = "mm"
    return out


def Rx5day(pr: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Maximum consecutive 5-day precipitation per period (mm)."""
    pr = _to_mm_per_day(pr)
    pr5 = _rolling_sum_5day(pr)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(pr5, freq=freq, reduce="max")
    out.name = "Rx5day"
    out.attrs["long_name"] = f"{period.capitalize()} maximum consecutive 5-day precipitation"
    out.attrs["units"] = "mm"
    return out


def SDII(pr: xr.DataArray, wet_day_thresh: float = 1.0,
         period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Simple Daily Intensity Index: mean precipitation on wet days (>= wet_day_thresh mm/day).
    """
    pr = _to_mm_per_day(pr)
    wet = _wetday_mask(pr, wet_day_thresh)
    wet_pr = pr.where(wet)
    freq = _coarsen_by_period(period)
    total = _resample_reduce(wet_pr.fillna(0.0), freq=freq, reduce="sum")
    ndays = _resample_reduce(wet.astype("int16"), freq=freq, reduce="sum")
    out = total / ndays
    out = out.where(ndays > 0)
    out.name = "SDII"
    out.attrs["long_name"] = f"{period.capitalize()} simple daily intensity index (mean on wet days ≥ {wet_day_thresh} mm)"
    out.attrs["units"] = "mm/day"
    return out


def R10mm(pr: xr.DataArray, threshold: float = 10.0,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Number of days with pr >= 10 mm (threshold configurable)."""
    pr = _to_mm_per_day(pr)
    mask = pr >= threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(mask.astype("int16"), freq=freq, reduce="sum")
    out.name = "R10mm"
    out.attrs["long_name"] = f"{period.capitalize()} count of days with pr ≥ {threshold} mm"
    out.attrs["units"] = "days"
    return out


def R20mm(pr: xr.DataArray, threshold: float = 20.0,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Number of days with pr >= 20 mm (threshold configurable)."""
    pr = _to_mm_per_day(pr)
    mask = pr >= threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(mask.astype("int16"), freq=freq, reduce="sum")
    out.name = "R20mm"
    out.attrs["long_name"] = f"{period.capitalize()} count of days with pr ≥ {threshold} mm"
    out.attrs["units"] = "days"
    return out


def PRCPTOT(pr: xr.DataArray, wet_day_thresh: float = 1.0,
            period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Total precipitation from wet days (pr ≥ wet_day_thresh) per period (mm)."""
    pr = _to_mm_per_day(pr)
    wet_pr = pr.where(pr >= wet_day_thresh)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(wet_pr.fillna(0.0), freq=freq, reduce="sum")
    out.name = "PRCPTOT"
    out.attrs["long_name"] = f"{period.capitalize()} total precipitation on wet days ≥ {wet_day_thresh} mm"
    out.attrs["units"] = "mm"
    return out


def _rXXpTOT(pr: xr.DataArray, q: float,
             base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
             window: int = 5,
             wet_day_thresh: float = 1.0,
             period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Total precipitation from days above the qth percentile of wet-day precipitation (R95pTOT, R99pTOT).
    """
    pr = _to_mm_per_day(pr)
    thr = _wet_percentile_thresholds_calendar_day(
        pr, base_period=base_period, q=q, window=window, wet_day_thresh=wet_day_thresh
    )
    # Days exceeding threshold (and wet)
    doy = pr["time"].dt.dayofyear
    doy = xr.where(doy == 366, 365, doy)
    t = thr.sel(doy=doy)
    mask = (pr >= wet_day_thresh) & (pr > t)
    contrib = pr.where(mask, 0.0)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(contrib, freq=freq, reduce="sum")
    out.name = f"R{int(q)}pTOT"
    out.attrs["long_name"] = f"{period.capitalize()} total precipitation from days > p{int(q)} (wet-day baseline)"
    out.attrs["units"] = "mm"
    return out


def R95pTOT(pr: xr.DataArray,
            base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
            window: int = 5,
            wet_day_thresh: float = 1.0,
            period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Total precipitation from days above 95th percentile of wet-day pr (mm)."""
    return _rXXpTOT(pr, q=95, base_period=base_period, window=window,
                    wet_day_thresh=wet_day_thresh, period=period)


def R99pTOT(pr: xr.DataArray,
            base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
            window: int = 5,
            wet_day_thresh: float = 1.0,
            period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Total precipitation from days above 99th percentile of wet-day pr (mm)."""
    return _rXXpTOT(pr, q=99, base_period=base_period, window=window,
                    wet_day_thresh=wet_day_thresh, period=period)


def CDD(pr: xr.DataArray, wet_day_thresh: float = 1.0,
        period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Consecutive Dry Days — maximum number of consecutive days with pr < wet_day_thresh.
    For each period, we return the **maximum run length** observed.
    """
    pr = _to_mm_per_day(pr)
    dry = pr < wet_day_thresh
    # Group by periods and compute max run length per group
    freq = _coarsen_by_period(period)
    grouped = dry.resample(time=freq)
    out = grouped.map(_max_run_length)
    out.name = "CDD"
    out.attrs["long_name"] = f"{period.capitalize()} maximum length of consecutive dry days (pr < {wet_day_thresh} mm)"
    out.attrs["units"] = "days"
    return _finalize_index_like(pr, out, "CDD", "days")


def CWD(pr: xr.DataArray, wet_day_thresh: float = 1.0,
        period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Consecutive Wet Days — maximum number of consecutive days with pr >= wet_day_thresh.
    For each period, we return the **maximum run length** observed.
    """
    pr = _to_mm_per_day(pr)
    wet = pr >= wet_day_thresh
    freq = _coarsen_by_period(period)
    grouped = wet.resample(time=freq)
    out = grouped.map(_max_run_length)
    out.name = "CWD"
    out.attrs["long_name"] = f"{period.capitalize()} maximum length of consecutive wet days (pr ≥ {wet_day_thresh} mm)"
    out.attrs["units"] = "days"
    return _finalize_index_like(pr, out, "CWD", "days")

# End of module
# --------------------------------------------------------------------------------------


# ===== Performance & grid-alignment overrides for R95pTOT, R99pTOT, CDD, CWD =====
import numpy as _np
import xarray as _xr

# Fallback: ensure finalizer exists
if "_finalize_index_like" not in globals():
    def _finalize_index_like(_src, _out, _name, _units):
        # order dims
        try:
            order = tuple(d for d in ("time","lat","lon") if d in _out.dims)
            if order:
                _out = _out.transpose(*order)
        except Exception:
            pass
        # align grid
        try:
            if "lat" in _out.dims and "lat" in _src.dims:
                _out = _out.assign_coords(lat=_src["lat"]).reindex(lat=_src["lat"])
            if "lon" in _out.dims and "lon" in _src.dims:
                _out = _out.assign_coords(lon=_src["lon"]).reindex(lon=_src["lon"])
        except Exception:
            pass
        # mask-all-NaN over time
        try:
            if "time" in _src.dims and "time" in _out.dims:
                _mask_obs = _src.notnull().any("time")
                _out = _out.where(_mask_obs)
        except Exception:
            pass
        # enforce chunks (time:1, lat:all, lon:all)
        try:
            chunks = {}
            if "time" in _out.dims: chunks["time"] = 1
            if "lat" in _out.dims: chunks["lat"] = -1
            if "lon" in _out.dims: chunks["lon"] = -1
            if chunks:
                _out = _out.chunk(chunks)
        except Exception:
            pass
        _out = _out.astype("float32")
        _out.name = _name
        _out.attrs["units"] = _units
        return _out

