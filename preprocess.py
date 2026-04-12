"""Data preprocessing utilities for drought resilience assessment.

This module implements the preprocessing logic described in:
Yuan et al. (2026) - integrating resistance, recovery, adaptability.

Main features
-------------
1. Optional regridding to target spatial resolution (default 0.05° in this project).
2. Monthly z-score standardization for dynamic variables.
3. GPP de-seasonalization (monthly climatology removal).
4. Per-pixel linear detrending for GPP.
5. Negative GPP anomaly mask generation using threshold -0.5 * std(detrended GPP).

Notes
-----
- All functions are xarray-first and work with dask-backed arrays.
- Large arrays are processed lazily when input datasets are chunked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import xarray as xr


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    target_resolution: float = 0.05
    time_freq: str = "MS"
    gpp_var: str = "gpp_anom"
    drought_var: str = "spei_anom"
    dynamic_vars: Sequence[str] = (
        "gpp_anom",
        "sm_anom",
        "t2m_anom",
        "rad_anom",
        "evi_anom",
        "gosif_anom",
        "spei_anom",
    )


def ensure_monthly_time(ds: xr.Dataset, freq: str = "MS") -> xr.Dataset:
    """Ensure dataset has monthly timestamps.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``time`` dimension.
    freq : str
        Pandas-style frequency string. Default monthly start ``MS``.
    """
    if "time" not in ds.dims:
        raise ValueError("Dataset must contain 'time' dimension.")

    # Resample only if time cadence is not monthly.
    inferred = xr.infer_freq(ds["time"].to_index())
    if inferred != freq:
        ds = ds.resample(time=freq).mean()
    return ds


def _build_target_grid(ds: xr.Dataset, resolution: float) -> Dict[str, np.ndarray]:
    """Build regular target lat/lon arrays from source extent."""
    lon_min = float(ds.lon.min())
    lon_max = float(ds.lon.max())
    lat_min = float(ds.lat.min())
    lat_max = float(ds.lat.max())

    lons = np.arange(lon_min, lon_max + resolution * 0.1, resolution)
    if ds.lat[0] > ds.lat[-1]:
        lats = np.arange(lat_max, lat_min - resolution * 0.1, -resolution)
    else:
        lats = np.arange(lat_min, lat_max + resolution * 0.1, resolution)
    return {"lon": lons, "lat": lats}


def regrid_to_resolution(ds: xr.Dataset, resolution: float = 0.05) -> xr.Dataset:
    """Interpolate dataset to regular target resolution with linear interpolation."""
    target = _build_target_grid(ds, resolution)
    return ds.interp(lat=target["lat"], lon=target["lon"], method="linear")


def zscore_standardize(
    ds: xr.Dataset,
    variables: Iterable[str],
    dim: str = "time",
    eps: float = 1e-6,
) -> xr.Dataset:
    """Apply z-score normalization to given variables along ``dim``.

    Returns a copied dataset where each listed variable is transformed to:
    (x - mean) / std
    """
    out = ds.copy()
    for var in variables:
        if var not in out:
            continue
        mean = out[var].mean(dim=dim, skipna=True)
        std = out[var].std(dim=dim, skipna=True)
        out[var] = (out[var] - mean) / xr.where(std < eps, np.nan, std)
    return out


def deseasonalize_monthly(gpp: xr.DataArray) -> xr.DataArray:
    """Remove monthly climatology from GPP."""
    clim = gpp.groupby("time.month").mean("time", skipna=True)
    return gpp.groupby("time.month") - clim


def detrend_1d(y: np.ndarray) -> np.ndarray:
    """Linear detrending for a 1D time series with NaN handling."""
    x = np.arange(y.size, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.full_like(y, np.nan, dtype=float)
    coeff = np.polyfit(x[mask], y[mask], 1)
    trend = coeff[0] * x + coeff[1]
    out = y - trend
    out[~mask] = np.nan
    return out


def detrend_per_pixel(da: xr.DataArray) -> xr.DataArray:
    """Detrend each pixel along ``time`` using ``apply_ufunc`` for scalability."""
    return xr.apply_ufunc(
        detrend_1d,
        da,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def compute_negative_anomaly_mask(
    gpp_detrended: xr.DataArray,
    factor: float = -0.5,
) -> xr.DataArray:
    """Mask where GPP is below ``factor * std`` per pixel.

    In the paper: threshold = -0.5 SD.
    """
    sd = gpp_detrended.std("time", skipna=True)
    threshold = factor * sd
    return gpp_detrended < threshold


def preprocess_dataset(ds: xr.Dataset, cfg: Optional[PreprocessConfig] = None) -> xr.Dataset:
    """Run full preprocessing and append derived fields to dataset.

    Derived variables:
    - gpp_deseason
    - gpp_detrended
    - gpp_neg_mask
    - gpp_neg_threshold
    """
    cfg = cfg or PreprocessConfig()

    ds = ensure_monthly_time(ds, freq=cfg.time_freq)
    ds = regrid_to_resolution(ds, resolution=cfg.target_resolution)
    ds = zscore_standardize(ds, cfg.dynamic_vars)

    gpp = ds[cfg.gpp_var]
    gpp_deseason = deseasonalize_monthly(gpp)
    gpp_detrended = detrend_per_pixel(gpp_deseason)

    neg_mask = compute_negative_anomaly_mask(gpp_detrended)
    neg_threshold = -0.5 * gpp_detrended.std("time", skipna=True)

    ds["gpp_deseason"] = gpp_deseason
    ds["gpp_detrended"] = gpp_detrended
    ds["gpp_neg_mask"] = neg_mask
    ds["gpp_neg_threshold"] = neg_threshold
    return ds
