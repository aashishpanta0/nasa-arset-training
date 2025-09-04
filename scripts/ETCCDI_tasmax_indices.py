"""
ETCCDI TX-based indices
-----------------------
Subroutines to compute common ETCCDI indices using **daily maximum temperature** (TX) only.
Designed to mirror a simple "helper functions" style like ETCCDI_pr_indices.py.

References
----------
ETCCDI list: https://etccdi.pacificclimate.org/list_27_indices.shtml

Requirements
------------
- xarray
- numpy
- pandas
- dask (optional but recommended for large arrays)

Input conventions
-----------------
- tasmax: xarray.DataArray of daily maximum temperature with a "time" dimension.
- Units should be degrees Celsius ("degC", "C", "celsius"). If units are Kelvin ("K"),
  they will be converted to Celsius automatically.
- Time coordinate should be daily and monotonic.

Implemented indices (TX-only)
-----------------------------
- TXx:   Monthly/annual maximum of daily maximum temperature (°C)
- TXn:   Monthly/annual minimum of daily maximum temperature (°C)
- SU25:  "Summer days" — count of days with TX > 25°C (per period)
- ID0:   "Ice days"    — count of days with TX < 0°C (per period)
- TX10p: Percentage of days when TX < 10th percentile (baseline, calendar-day method)
- TX90p: Percentage of days when TX > 90th percentile (baseline, calendar-day method)
- WSDI:  Warm Spell Duration Index — number of days in spells of ≥6 consecutive days
         with TX > p90 (calendar-day thresholds; reported per period)

Notes on percentile-based indices
---------------------------------
Following the ETCCDI approach, percentiles are computed for each calendar day using a 5-day
moving window (±2 days around the day-of-year) over a user-specified baseline period
(e.g., 1961–1990). The thresholds are then applied to the full analysis period.

These routines are intentionally self-contained and avoid external climate index libraries.
For production use and full ETCCDI fidelity, consider using xclim (https://xclim.readthedocs.io/).
"""


from __future__ import annotations

__DISCLAIMER__ = "ALL INDICES PRODUCED BY THESE SCRIPTS/MODULES ARE EXPERIMENTAL. USERS MUST REVIEW THE CODE AND VALIDATE RESULTS AGAINST KNOWN REFERENCES BEFORE ANY OPERATIONAL OR DECISION SUPPORT USE. THERE MAY BE ERRORS."

import numpy as np
import pandas as pd
import xarray as xr

def _finalize_index_like(_src, _out, _name, _units):
    import numpy as _np
    import xarray as _xr
    # order dims for consistency
    try:
        order = tuple(d for d in ("time","lat","lon") if d in _out.dims)
        if order:
            _out = _out.transpose(*order)
    except Exception:
        pass

    # align coordinates to source grid
    try:
        if "lat" in _out.dims and "lat" in _src.dims:
            _out = _out.assign_coords(lat=_src["lat"])
        if "lon" in _out.dims and "lon" in _src.dims:
            _out = _out.assign_coords(lon=_src["lon"])
        if "lat" in _out.dims and "lat" in _src.dims:
            _out = _out.reindex(lat=_src["lat"])
        if "lon" in _out.dims and "lon" in _src.dims:
            _out = _out.reindex(lon=_src["lon"])
    except Exception:
        pass

    # mask-all-NaN over time
    try:
        if "time" in _src.dims and "time" in _out.dims:
            _mask_obs = _src.notnull().any("time")
            _out = _out.where(_mask_obs)
    except Exception:
        pass

    # force full spatial chunking and time chunk=1
    try:
        chunks = {}
        if "time" in _out.dims:
            chunks["time"] = 1
        if "lat" in _out.dims:
            chunks["lat"] = -1
        if "lon" in _out.dims:
            chunks["lon"] = -1
        if chunks:
            _out = _out.chunk(chunks)
    except Exception:
        pass

    _out = _out.astype("float32")
    _out.name = _name
    _out.attrs["units"] = _units
    return _out

# --- Robust DOY helper (works with numpy datetime64 and cftime; falls back gracefully) ---
def _dayofyear_coord(_da):
    try:
        return _da["time"].dt.dayofyear
    except Exception:
        import numpy as _np
        import xarray as _xr
        t = _da["time"]
        vals = t.values
        try:
            # cftime or datetime-like objects with .dayofyear or timetuple()
            doy = _np.array([
                (getattr(v, "dayofyear") if hasattr(v, "dayofyear") else (v.timetuple().tm_yday if hasattr(v, "timetuple") else _np.nan))
                for v in vals
            ])
            return _xr.DataArray(doy, dims=("time",), coords={"time": t})
        except Exception as _e:
            # Last resort: try decode_cf
            try:
                _ds_tmp = _xr.Dataset(coords={"time": t})
                _ds_dec = _xr.decode_cf(_ds_tmp)
                return _ds_dec["time"].dt.dayofyear
            except Exception:
                raise TypeError("time coordinate is not datetime-like; ensure decode_times=True or valid CF time attrs") from _e
from typing import Tuple, Optional, Literal

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Ensure temperature is in degrees Celsius based on the 'units' attribute."""
    units = (str(da.attrs.get("units", "")).lower() or "").replace("°", "")
    if units in {"k", "kelvin"}:
        out = da - 273.15
        out.attrs["units"] = "degC"
        return out
    # Common Celsius strings
    if units in {"degc", "c", "celsius"}:
        return da
    # Unknown: assume input already in Celsius but standardize attribute
    out = da.copy()
    out.attrs["units"] = "degC"
    return out


def _resample_reduce(da: xr.DataArray, freq: str, reduce: str) -> xr.DataArray:
    """
    Generic reducer for a given resampling frequency.

    Parameters
    ----------
    da : xr.DataArray
        Daily data with time dimension.
    freq : str
        Resampling frequency. Common choices:
        - "D"  : daily (no resample)
        - "MS" : month start labels
        - "YS" : year start labels
        - "QS-DEC": seasonal (DJF,MAM,JJA,SON) with seasons ending in Nov (labels at season start)
    reduce : {"max","min","sum","mean"}
    """
    if freq == "D":
        r = da
    else:
        r = da.resample(time=freq)

    if reduce == "max":
        out = r.max(keep_attrs=True)
    elif reduce == "min":
        out = r.min(keep_attrs=True)
    elif reduce == "sum":
        out = r.sum(keep_attrs=True)
    elif reduce == "mean":
        out = r.mean(keep_attrs=True)
    else:
        raise ValueError(f"Unsupported reduce={reduce}")

    return out


def _coarsen_by_period(da: xr.DataArray, period: Literal["monthly","annual","seasonal"]) -> str:
    """
    Map simple period keywords to resample frequencies.

    Returns the resample frequency string to be used with .resample(time=freq).
    For seasonal, uses meteorological seasons with labels at season start (Dec-based quarters).
    """
    if period == "monthly":
        return "MS"
    if period == "annual":
        return "YS"
    if period == "seasonal":
        # Seasons defined as: DJF, MAM, JJA, SON — label at season start (Dec-based quarter)
        # Using quarter frequency with year ending in Nov (DEC anchor) puts DJF together.
        return "QS-DEC"
    raise ValueError("period must be one of {'monthly','annual','seasonal'}")


def _percentile_thresholds_calendar_day(
    da: xr.DataArray,
    base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
    q: float = 90.0,
    window: int = 5,
) -> xr.DataArray:
    """
    Compute calendar-day percentile thresholds with a moving window over a baseline period.

    Parameters
    ----------
    da : xr.DataArray
        Daily data (time dimension) in Celsius.
    base_period : (start, end)
        Baseline period inclusive bounds (ISO date strings).
    q : float
        Percentile (0-100).
    window : int
        Odd integer window size (default 5) centered on each calendar day.

    Returns
    -------
    xr.DataArray
        Thresholds indexed by day-of-year (1..365) with a "doy" coordinate.
        Leap day (Feb 29) is mapped to 365 (Dec 31 window) for robustness.
    """
    if window % 2 != 1:
        raise ValueError("window must be an odd integer (e.g., 5, 7, 11)")

    base = da.sel(time=slice(base_period[0], base_period[1]))
    if base.time.size == 0:
        raise ValueError("Baseline period selection is empty. Check base_period and data time range.")

    # Day-of-year (1..366); map 366->365
    doy = _dayofyear_coord(base)
    doy = xr.where(doy == 366, 365, doy)
    base = base.assign_coords(doy=("time", doy))

    half = window // 2
    doys = np.arange(1, 366)  # 1..365
    # Prepare output with same spatial dims (if any)
    sample = base.isel(time=0).drop_vars([v for v in ["doy"] if v in base.coords])
    shp = (365,) + tuple(sample.shape)
    thresh = np.empty(shp, dtype=float)

    # Loop on CPU; works with/without dask (chunking along time still helps IO)
    # To reduce memory, we compute quantiles on the fly.
    for i, d in enumerate(doys):
        lo = d - half
        hi = d + half
        window_doys = ((np.arange(lo, hi + 1) - 1) % 365) + 1
        sel = base.where(base["doy"].isin(window_doys), drop=True)
        # Percentile across time axis
        # Use xarray's quantile when available (keeps dims), but here we fall back to numpy for broad compatibility.
        arr = sel.transpose(..., "time").values  # ensure last dim is time
        # last axis is time
        thresh[i, ...] = np.nanpercentile(arr, q, axis=-1)

    out = xr.DataArray(
        thresh,
        dims=("doy",) + sample.dims,
        coords={"doy": doys, **{k: v for k in sample.coords.keys()}},
        attrs={"description": f"{q}th percentile threshold by calendar day", "units": da.attrs.get("units", "degC")},
    )
    return out


def _apply_doy_threshold(
    da: xr.DataArray,
    thresh_doy: xr.DataArray,
    op: str = ">",
) -> xr.DataArray:
    """
    Compare daily values to DOY-indexed thresholds.

    Parameters
    ----------
    da : xr.DataArray
        Daily data with time coordinate.
    thresh_doy : xr.DataArray
        Thresholds by 'doy' coordinate (1..365).
    op : {">", "<", ">=", "<="}

    Returns
    -------
    xr.DataArray
        Boolean mask over time (and space) where condition holds.
    """
    doy = da["time"].dt.dayofyear
    doy = xr.where(doy == 366, 365, doy)  # map leap day to 365
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


def _count_runs(mask: xr.DataArray, run_length: int = 6) -> xr.DataArray:
    """
    Return an array with the total number of days that are part of runs
    of consecutive True values having length >= run_length.

    This is used for WSDI where we count the days inside warm spells.
    The output keeps the same shape as mask, and we later aggregate by period.
    """
    # We compute run lengths along 'time' per grid cell using a simple approach:
    # - Identify boundaries where mask changes
    # - Compute run lengths via cumsum trick
    # For xarray compatibility, we work with numpy via apply_ufunc.
    def _run_days_1d(b):
        if b.size == 0:
            return b.astype(int)
        # Convert True/False to 1/0
        x = b.astype(np.int8)
        # Positions where a run starts (x=1 and previous 0)
        start = np.where((x == 1) & (np.roll(x, 1) == 0))[0]
        start = start[start != 0] if x[0] == 1 else start
        if x[0] == 1:
            start = np.r_[0, start]
        # Positions where a run ends (x=1 and next 0)
        end = np.where((x == 1) & (np.roll(x, -1) == 0))[0]
        end = end[end != x.size - 1] if x[-1] == 1 else end
        if x[-1] == 1:
            end = np.r_[end, x.size - 1]
        # Mark days belonging to runs >= run_length
        out = np.zeros_like(x, dtype=np.int8)
        for s, e in zip(start, end):
            if (e - s + 1) >= run_length:
                out[s:e+1] = 1
        return out

    run_days = xr.apply_ufunc(
        _run_days_1d,
        mask,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int8],
    )
    return run_days

#-------------------------------------------------


def TXx(tasmax: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Maximum of daily maximum temperature (°C) per period (monthly/annual/seasonal)."""
    tx = _to_celsius(tasmax)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(tasmax, freq=freq, reduce="max")
    out.name = "TXx"
    out.attrs["long_name"] = f"{period.capitalize()} maximum of daily maximum temperature"
    out.attrs["units"] = "degC"
    return out


def TXn(tasmax: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Minimum of daily maximum temperature (°C) per period (monthly/annual/seasonal)."""
    tx = _to_celsius(tasmax)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(tasmax, freq=freq, reduce="min")
    out.name = "TXn"
    out.attrs["long_name"] = f"{period.capitalize()} minimum of daily maximum temperature"
    out.attrs["units"] = "degC"
    return out


def SU25(tasmax: xr.DataArray, threshold: float = 25.0,
         period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Summer days: number of days when TX > threshold (default 25°C).

    Parameters
    ----------
    tx : xr.DataArray
        Daily max temperature.
    threshold : float
        Degrees Celsius.
    period : {"monthly","annual","seasonal"}
        Aggregation period.
    """
    tx = _to_celsius(tasmax)
    hot = tasmax > threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(hot.astype("int16"), freq=freq, reduce="sum")
    out.name = "SU25"
    out.attrs["long_name"] = f"{period.capitalize()} count of days with TX > {threshold}°C"
    out.attrs["units"] = "days"
    return out


def ID0(tasmax: xr.DataArray, threshold: float = 0.0,
        period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Ice days: number of days when TX < threshold (default 0°C).

    Parameters
    ----------
    tx : xr.DataArray
        Daily max temperature.
    threshold : float
        Degrees Celsius.
    period : {"monthly","annual","seasonal"}
        Aggregation period.
    """
    tx = _to_celsius(tasmax)
    cold = tasmax < threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(cold.astype("int16"), freq=freq, reduce="sum")
    out.name = "ID0"
    out.attrs["long_name"] = f"{period.capitalize()} count of days with TX < {threshold}°C"
    out.attrs["units"] = "days"
    return out


def TX_percentile_frequency(
    tasmax: xr.DataArray,
    q: float,
    base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
    window: int = 5,
    compare: Literal[">", "<"] = ">",
    period: Literal["monthly","annual","seasonal"]="annual",
    pct: bool = True,
) -> xr.DataArray:
    """
    Generic calendar-day percentile frequency (% or days) against baseline thresholds.

    Examples:
    - TX90p : q=90, compare=">", pct=True
    - TX10p : q=10, compare="<", pct=True
    """
    tx = _to_celsius(tasmax)
    thr = _percentile_thresholds_calendar_day(tasmax, base_period=base_period, q=q, window=window)
    mask = _apply_doy_threshold(tasmax, thr, op=compare)
    # Count days per period
    freq = _coarsen_by_period(period)
    days = _resample_reduce(mask.astype("int16"), freq=freq, reduce="sum")
    days.name = f"TX{int(q)}{ 'p' if pct else 'd' }"
    # Compute percentage relative to number of valid days in each period
    if pct:
        # Valid days: count non-NaN in original data
        valid = _resample_reduce((~np.isnan(tasmax)).astype("int16"), freq=freq, reduce="sum")
        out = 100.0 * days / valid
        out = out.where(valid > 0)  # avoid div-by-zero
        out.attrs["units"] = "%"
        out.attrs["long_name"] = f"{period.capitalize()} percentage of days with TX {compare} p{int(q)}"
        return out
    else:
        days.attrs["units"] = "days"
        days.attrs["long_name"] = f"{period.capitalize()} count of days with TX {compare} p{int(q)}"
        return days


def TX90p(tasmax: xr.DataArray,
          base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
          window: int = 5,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Percentage of days when TX > 90th percentile (calendar-day thresholds)."""
    return TX_percentile_frequency(tasmax, q=90, base_period=base_period, window=window,
                                   compare=">", period=period, pct=True)


def TX10p(tasmax: xr.DataArray,
          base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
          window: int = 5,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Percentage of days when TX < 10th percentile (calendar-day thresholds)."""
    return TX_percentile_frequency(tasmax, q=10, base_period=base_period, window=window,
                                   compare="<", period=period, pct=True)


def WSDI(tasmax: xr.DataArray,
         base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
         window: int = 5,
         min_run: int = 6,
         period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Warm Spell Duration Index — number of days in spells of >= min_run consecutive days
    with TX above the calendar-day 90th percentile threshold (baseline windowed).

    Output is aggregated by the requested period (default annual).

    Notes:
    - We follow ETCCDI definition using p90 and 6-day minimum spell length by default.
    """
    tx = _to_celsius(tasmax)
    thr90 = _percentile_thresholds_calendar_day(tasmax, base_period=base_period, q=90, window=window)
    hot = _apply_doy_threshold(tasmax, thr90, op=">")
    run_days = _count_runs(hot, run_length=min_run)  # 1 for days inside qualifying spells
    freq = _coarsen_by_period(period)
    out = _resample_reduce(run_days.astype("int16"), freq=freq, reduce="sum")
    out.name = "WSDI"
    out.attrs["long_name"] = f"{period.capitalize()} warm spell duration index (days), runs≥{min_run} with TX>p90"
    out.attrs["units"] = "days"
    return _finalize_index_like(tasmax, out, "WSDI", "days")


# End of module
# --------------------------------------------------------------------------------------

# ===== Overrides appended to ensure consistent behavior and mask-all-NaN handling =====
import xarray as _xr
import numpy as _np

def _to_celsius_if_k(da):
    units = str(da.attrs.get("units","")).lower()
    if units in ("k","kelvin"):
        da = da - 273.15
        da.attrs["units"] = "degC"
    return da

def _mask_all_nan(time_series_da, out_da):
    mask = time_series_da.notnull().any(dim="time")
    return out_da.where(mask)

def _drop_quantile(da):
    # Remove lingering 'quantile' coordinate/dim that can cause Dataset merge conflicts
    if "quantile" in getattr(da, "dims", ()):
        da = da.isel(quantile=0).drop_vars("quantile")
    elif "quantile" in getattr(da, "coords", {}):
        da = da.reset_coords("quantile", drop=True)
    return da

def _calendar_day_threshold(ta, q=90, base_period=("1961-01-01","1990-12-31"), window=5):
    ta = ta.chunk({'time': -1}) if getattr(ta, 'chunks', None) else ta
    base = ta.sel(time=slice(base_period[0], base_period[1]))
    day = _dayofyear_coord(base)
    day = _xr.where(day==366, 365, day)
    if day.size == 0:
        # fallback protection
        base = ta
        day = _dayofyear_coord(base)
        day = _xr.where(day==366, 365, day)
    thr = base.groupby(day).quantile(q/100.0, dim="time", skipna=True)
    if "quantile" in thr.dims:
        thr = thr.sel(quantile=q/100.0, drop=True)
    try:
        if "group" in thr.coords:
            thr = thr.rename({"group":"dayofyear"})
    except Exception:
        pass
    if window and window > 1:
        pad = window//2
        thr = _xr.concat([thr.isel(dayofyear=slice(-pad,None)), thr, thr.isel(dayofyear=slice(0,pad))], dim="dayofyear")
        thr = thr.rolling(dayofyear=window, center=True, min_periods=1).mean().isel(dayofyear=slice(pad, pad+365))
    return thr

def _align_threshold(ta, thr):
    day_all = _dayofyear_coord(ta)
    day_all = _xr.where(day_all==366, 365, day_all)
    return thr.sel(dayofyear=day_all)

def _count_spell_days_1d(mask, min_len=6):
    b = _np.asarray(mask, dtype=bool)
    if b.size == 0:
        return _np.int16(0)
    total = 0
    run = 0
    for v in b:
        if v:
            run += 1
        else:
            if run >= min_len:
                total += run
            run = 0
    if run >= min_len:
        total += run
    return _np.int16(total)

def _spell_days(mask, min_len=6):
    mask = mask.chunk({'time': -1}) if getattr(mask, 'chunks', None) else mask
    return mask.resample(time="YS").map(
        lambda x: _xr.apply_ufunc(
            _count_spell_days_1d, x,
            input_core_dims=[["time"]], output_core_dims=[[]],
            vectorize=True, dask="parallelized",
            output_dtypes=[_np.int16],
            dask_gufunc_kwargs={'allow_rechunk': True},
            kwargs={"min_len": min_len}
        )
    )

# The following overrides accept 'period' and compute annual stats via 'YS' resample.
def TXx(tasmax, period="annual"):
    tasmax = _to_celsius_if_k(tasmax)
    out = tasmax.resample(time="YS").max("time")
    out.name = "TXx"; out.attrs["units"] = "degC"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def TXn(tasmax, period="annual"):
    tasmax = _to_celsius_if_k(tasmax)
    out = tasmax.resample(time="YS").min("time")
    out.name = "TXn"; out.attrs["units"] = "degC"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def SU25(tasmax, period="annual"):
    tasmax = _to_celsius_if_k(tasmax)
    out = (tasmax > 25.0).resample(time="YS").sum("time").astype("int16")
    out.name = "SU25"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def ID0(tasmax, period="annual"):
    tasmax = _to_celsius_if_k(tasmax)
    out = (tasmax <  0.0).resample(time="YS").sum("time").astype("int16")
    out.name = "ID0"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def TX90p(tasmax, period="annual", base_period=("1961-01-01","1990-12-31"), window=5):
    tasmax = _to_celsius_if_k(tasmax)
    thr = _calendar_day_threshold(tasmax, q=90, base_period=base_period, window=window)
    t90a = _align_threshold(tasmax, thr)
    mask = (tasmax > t90a).chunk({'time': -1}) if getattr(tasmax, 'chunks', None) else (tasmax > t90a)
    days = mask.resample(time="YS").sum("time")
    total = mask.resample(time="YS").count("time").clip(min=1)
    out = (100.0 * days / total).astype("float32")
    out.name = "TX90p"; out.attrs["units"] = "%"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def TX10p(tasmax, period="annual", base_period=("1961-01-01","1990-12-31"), window=5):
    tasmax = _to_celsius_if_k(tasmax)
    thr = _calendar_day_threshold(tasmax, q=10, base_period=base_period, window=window)
    t10a = _align_threshold(tasmax, thr)
    mask = (tasmax < t10a).chunk({'time': -1}) if getattr(tasmax, 'chunks', None) else (tasmax < t10a)
    days = mask.resample(time="YS").sum("time")
    total = mask.resample(time="YS").count("time").clip(min=1)
    out = (100.0 * days / total).astype("float32")
    out.name = "TX10p"; out.attrs["units"] = "%"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def WSDI(tasmax, period="annual", base_period=("1961-01-01","1990-12-31"), window=5, min_length=6):
    tasmax = _to_celsius_if_k(tasmax)
    thr = _calendar_day_threshold(tasmax, q=90, base_period=base_period, window=window)
    t90a = _align_threshold(tasmax, thr)
    mask = (tasmax > t90a).chunk({'time': -1}) if getattr(tasmax, 'chunks', None) else (tasmax > t90a)
    out = _spell_days(mask, min_len=min_length).astype("int16")
    out.name = "WSDI"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmax, out)

def TNx(tasmin, period="annual"):
    tasmin = _to_celsius_if_k(tasmin)
    out = tasmin.resample(time="YS").max("time")
    out.name = "TNx"; out.attrs["units"] = "degC"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def TNn(tasmin, period="annual"):
    tasmin = _to_celsius_if_k(tasmin)
    out = tasmin.resample(time="YS").min("time")
    out.name = "TNn"; out.attrs["units"] = "degC"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def TR20(tasmin, period="annual"):
    tasmin = _to_celsius_if_k(tasmin)
    out = (tasmin > 20.0).resample(time="YS").sum("time").astype("int16")
    out.name = "TR20"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def FD0(tasmin, period="annual"):
    tasmin = _to_celsius_if_k(tasmin)
    out = (tasmin <  0.0).resample(time="YS").sum("time").astype("int16")
    out.name = "FD0"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def TN90p(tasmin, period="annual", base_period=("1961-01-01","1990-12-31"), window=5):
    tasmin = _to_celsius_if_k(tasmin)
    thr = _calendar_day_threshold(tasmin, q=90, base_period=base_period, window=window)
    t90a = _align_threshold(tasmin, thr)
    mask = (tasmin > t90a).chunk({'time': -1}) if getattr(tasmin, 'chunks', None) else (tasmin > t90a)
    days = mask.resample(time="YS").sum("time")
    total = mask.resample(time="YS").count("time").clip(min=1)
    out = (100.0 * days / total).astype("float32")
    out.name = "TN90p"; out.attrs["units"] = "%"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def TN10p(tasmin, period="annual", base_period=("1961-01-01","1990-12-31"), window=5):
    tasmin = _to_celsius_if_k(tasmin)
    thr = _calendar_day_threshold(tasmin, q=10, base_period=base_period, window=window)
    t10a = _align_threshold(tasmin, thr)
    mask = (tasmin < t10a).chunk({'time': -1}) if getattr(tasmin, 'chunks', None) else (tasmin < t10a)
    days = mask.resample(time="YS").sum("time")
    total = mask.resample(time="YS").count("time").clip(min=1)
    out = (100.0 * days / total).astype("float32")
    out.name = "TN10p"; out.attrs["units"] = "%"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)

def CSDI(tasmin, period="annual", base_period=("1961-01-01","1990-12-31"), window=5, min_length=6):
    tasmin = _to_celsius_if_k(tasmin)
    thr = _calendar_day_threshold(tasmin, q=10, base_period=base_period, window=window)
    t10a = _align_threshold(tasmin, thr)
    mask = (tasmin < t10a).chunk({'time': -1}) if getattr(tasmin, 'chunks', None) else (tasmin < t10a)
    out = _spell_days(mask, min_len=min_length).astype("int16")
    out.name = "CSDI"; out.attrs["units"] = "days"
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    out = _drop_quantile(out)
    return _mask_all_nan(tasmin, out)
