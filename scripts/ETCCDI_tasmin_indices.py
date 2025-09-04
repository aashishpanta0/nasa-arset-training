"""
ETCCDI TN-based indices (tasmin)
---------------------------------
Subroutines to compute common ETCCDI indices using **daily minimum temperature** as input
(`tasmin`: xarray.DataArray with a "time" dimension). The style mirrors simple helper
functions (like ETCCDI_pr_indices.py and ETCCDI_tx_indices.py).

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
- tasmin: xarray.DataArray of daily minimum temperature with a "time" dimension.
- Units should be degrees Celsius ("degC", "C", "celsius"). If units are Kelvin ("K"),
  they will be converted to Celsius automatically.
- Time coordinate should be daily and monotonic.

Implemented indices (tasmin-only)
---------------------------------
- TNx:   Period max of daily minimum temperature (°C) — monthly/annual/seasonal
- TNn:   Period min of daily minimum temperature (°C)
- TR20:  "Tropical nights" — count of days with TN > 20°C (per period; threshold configurable)
- FD0:   "Frost days"      — count of days with TN < 0°C (per period; threshold configurable)
- TN10p: Percentage of days when TN < 10th percentile (baseline, calendar-day method)
- TN90p: Percentage of days when TN > 90th percentile (baseline, calendar-day method)
- CSDI:  Cold Spell Duration Index — number of days in spells of ≥6 consecutive days
         with TN < p10 (calendar-day thresholds; reported per period)

Percentile-based indices follow ETCCDI:
- Calendar-day thresholds using a 5-day moving window (±2 days) over a baseline period (default 1961–1990).
- Thresholds are then applied to the full analysis period.
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
from typing import Tuple, Literal

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
    if units in {"degc", "c", "celsius"}:
        return da
    out = da.copy()
    out.attrs["units"] = "degC"
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


def _percentile_thresholds_calendar_day(
    da: xr.DataArray,
    base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
    q: float = 90.0,
    window: int = 5,
) -> xr.DataArray:
    """
    Compute calendar-day percentile thresholds with a moving window over a baseline period.

    Returns thresholds indexed by day-of-year (1..365). Leap day is mapped to 365.
    """
    if window % 2 != 1:
        raise ValueError("window must be an odd integer (e.g., 5, 7, 11)")

    base = da.sel(time=slice(base_period[0], base_period[1]))
    if base.time.size == 0:
        raise ValueError("Baseline period selection is empty. Check base_period and data time range.")

    doy = _dayofyear_coord(base)
    doy = xr.where(doy == 366, 365, doy)  # map leap day to 365
    base = base.assign_coords(doy=("time", doy))

    half = window // 2
    doys = np.arange(1, 366)
    sample = base.isel(time=0).drop_vars([v for v in ["doy"] if v in base.coords])
    shp = (365,) + tuple(sample.shape)
    thresh = np.empty(shp, dtype=float)

    for i, d in enumerate(doys):
        lo, hi = d - half, d + half
        window_doys = ((np.arange(lo, hi + 1) - 1) % 365) + 1
        sel = base.where(base["doy"].isin(window_doys), drop=True)
        arr = sel.transpose(..., "time").values
        thresh[i, ...] = np.nanpercentile(arr, q, axis=-1)

    out = xr.DataArray(
        thresh,
        dims=("doy",) + sample.dims,
        coords={"doy": doys, **{k: v for k in sample.coords.keys()}},
        attrs={"description": f"{q}th percentile threshold by calendar day", "units": da.attrs.get("units", "degC")},
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


def _count_runs(mask: xr.DataArray, run_length: int = 6) -> xr.DataArray:
    """
    Return an array marking days that are part of runs of consecutive True values
    having length >= run_length. Used for CSDI.
    """
    def _run_days_1d(b):
        if b.size == 0:
            return b.astype(np.int8)
        x = b.astype(np.int8)
        # find starts
        start = np.where((x == 1) & (np.roll(x, 1) == 0))[0]
        start = start[start != 0] if x[0] == 1 else start
        if x[0] == 1:
            start = np.r_[0, start]
        # find ends
        end = np.where((x == 1) & (np.roll(x, -1) == 0))[0]
        end = end[end != x.size - 1] if x[-1] == 1 else end
        if x[-1] == 1:
            end = np.r_[end, x.size - 1]
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

#--------------------------------------------

def TNx(tasmin: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Maximum of daily minimum temperature (°C) per period (monthly/annual/seasonal)."""
    tasmin = _to_celsius(tasmin)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(tasmin, freq=freq, reduce="max")
    out.name = "TNx"
    out.attrs["long_name"] = f"{period.capitalize()} maximum of daily minimum temperature"
    out.attrs["units"] = "degC"
    return out


def TNn(tasmin: xr.DataArray, period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Minimum of daily minimum temperature (°C) per period (monthly/annual/seasonal)."""
    tasmin = _to_celsius(tasmin)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(tasmin, freq=freq, reduce="min")
    out.name = "TNn"
    out.attrs["long_name"] = f"{period.capitalize()} minimum of daily minimum temperature"
    out.attrs["units"] = "degC"
    return out


def TR20(tasmin: xr.DataArray, threshold: float = 20.0,
         period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Tropical nights: number of days when TN > threshold (default 20°C).
    """
    tasmin = _to_celsius(tasmin)
    warm_nights = tasmin > threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(warm_nights.astype("int16"), freq=freq, reduce="sum")
    out.name = "TR20"
    out.attrs["long_name"] = f"{period.capitalize()} count of nights with TN > {threshold}°C"
    out.attrs["units"] = "days"
    return out


def FD0(tasmin: xr.DataArray, threshold: float = 0.0,
        period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Frost days: number of days when TN < threshold (default 0°C).
    """
    tasmin = _to_celsius(tasmin)
    frost = tasmin < threshold
    freq = _coarsen_by_period(period)
    out = _resample_reduce(frost.astype("int16"), freq=freq, reduce="sum")
    out.name = "FD0"
    out.attrs["long_name"] = f"{period.capitalize()} count of days with TN < {threshold}°C"
    out.attrs["units"] = "days"
    return out


def TN_percentile_frequency(
    tasmin: xr.DataArray,
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
      - TN90p : q=90, compare=">", pct=True
      - TN10p : q=10, compare="<", pct=True
    """
    tasmin = _to_celsius(tasmin)
    thr = _percentile_thresholds_calendar_day(tasmin, base_period=base_period, q=q, window=window)
    mask = _apply_doy_threshold(tasmin, thr, op=compare)
    freq = _coarsen_by_period(period)
    days = _resample_reduce(mask.astype("int16"), freq=freq, reduce="sum")
    days.name = f"TN{int(q)}{ 'p' if pct else 'd' }"
    if pct:
        valid = _resample_reduce((~np.isnan(tasmin)).astype("int16"), freq=freq, reduce="sum")
        out = 100.0 * days / valid
        out = out.where(valid > 0)
        out.attrs["units"] = "%"
        out.attrs["long_name"] = f"{period.capitalize()} percentage of days with TN {compare} p{int(q)}"
        return out
    else:
        days.attrs["units"] = "days"
        days.attrs["long_name"] = f"{period.capitalize()} count of days with TN {compare} p{int(q)}"
        return days


def TN90p(tasmin: xr.DataArray,
          base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
          window: int = 5,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Percentage of days when TN > 90th percentile (calendar-day thresholds)."""
    return TN_percentile_frequency(tasmin, q=90, base_period=base_period, window=window,
                                   compare=">", period=period, pct=True)


def TN10p(tasmin: xr.DataArray,
          base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
          window: int = 5,
          period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """Percentage of days when TN < 10th percentile (calendar-day thresholds)."""
    return TN_percentile_frequency(tasmin, q=10, base_period=base_period, window=window,
                                   compare="<", period=period, pct=True)


def CSDI(tasmin: xr.DataArray,
         base_period: Tuple[str, str] = ("1961-01-01", "1990-12-31"),
         window: int = 5,
         min_run: int = 6,
         period: Literal["monthly","annual","seasonal"]="annual") -> xr.DataArray:
    """
    Cold Spell Duration Index — number of days in spells of >= min_run consecutive days
    with TN below the calendar-day 10th percentile threshold.

    Output is aggregated by the requested period (default annual).
    """
    tasmin = _to_celsius(tasmin)
    thr10 = _percentile_thresholds_calendar_day(tasmin, base_period=base_period, q=10, window=window)
    cold = _apply_doy_threshold(tasmin, thr10, op="<")
    run_days = _count_runs(cold, run_length=min_run)
    freq = _coarsen_by_period(period)
    out = _resample_reduce(run_days.astype("int16"), freq=freq, reduce="sum")
    out.name = "CSDI"
    out.attrs["long_name"] = f"{period.capitalize()} cold spell duration index (days), runs≥{min_run} with TN<p10"
    out.attrs["units"] = "days"
    return _finalize_index_like(tasmin, out, "CSDI", "days")

# --------------------------------------------------------------------------------------
# Convenience: select season
# --------------------------------------------------------------------------------------

def select_season(da: xr.DataArray, season: Literal["DJF","MAM","JJA","SON"]) -> xr.DataArray:
    """Return a seasonal subset of a daily or coarsened series using season labels."""
    if "season" in da.coords and "time" not in da.dims:
        return da.sel(season=season)
    months = {"DJF":[12,1,2], "MAM":[3,4,5], "JJA":[6,7,8], "SON":[9,10,11]}[season]
    return da.where(da["time"].dt.month.isin(months), drop=True)


# End of module

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
    out = _drop_quantile(out)
    
    return _mask_all_nan(tasmin, out)
