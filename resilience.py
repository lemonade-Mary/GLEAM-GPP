"""Resilience metric calculations.

Implements event-level and pixel-level metrics:
- Normalized Loss
- Resistance Time
- Recovery Time
- Resistance, Recovery
- Adaptability (A_prev / A_i)
- Entropy-weighted integrated Resilience
- AR(1) validation series
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr


EPS = 1e-8


def minmax_01(s: pd.Series) -> pd.Series:
    """Normalize series to [0,1] with safe handling of constants."""
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or np.isclose(hi - lo, 0):
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def compute_event_metrics(
    events: pd.DataFrame,
    ds: xr.Dataset,
    gpp_detrended_var: str = "gpp_detrended",
    drought_var: str = "spei_anom",
) -> pd.DataFrame:
    """Compute event-level resistance, recovery and adaptability ingredients."""
    if events.empty:
        return events.copy()

    out = events.copy()
    out["normalized_loss"] = np.nan
    out["resistance_time"] = np.nan
    out["recovery_time"] = np.nan
    out["resistance_raw"] = np.nan
    out["recovery_raw"] = np.nan
    out["Ai"] = np.nan
    out["Aprev"] = np.nan
    out["adaptability_raw"] = np.nan

    # Group by pixel for A_prev historical averaging.
    grouped = out.groupby(["lat", "lon"], sort=False)

    for (lat, lon), idxs in grouped.groups.items():
        pixel_events = out.loc[idxs].sort_values("start_idx")
        prev_ai = []

        for idx, row in pixel_events.iterrows():
            yi = int(np.argmin(np.abs(ds.lat.values - row.lat)))
            xi = int(np.argmin(np.abs(ds.lon.values - row.lon)))

            t0 = int(row.start_idx)
            t1 = int(row.end_idx)
            tmin = int(row.min_idx)

            gpp_ts = ds[gpp_detrended_var][:, yi, xi].values
            mcdi_ts = ds[drought_var][:, yi, xi].values

            threshold = -0.5 * np.nanstd(gpp_ts)
            min_val = float(np.nanmin(gpp_ts[t0 : t1 + 1]))
            normalized_loss = threshold - min_val

            # Resistance time: from onset to minimum.
            resistance_time = max(tmin - t0 + 1, 1)

            # Recovery time: from minimum until GPP returns above threshold.
            rec_end = t1
            for t in range(tmin, len(gpp_ts)):
                if np.isfinite(gpp_ts[t]) and gpp_ts[t] >= threshold:
                    rec_end = t
                    break
            recovery_time = max(rec_end - tmin + 1, 1)

            resistance = resistance_time / (normalized_loss + EPS)
            recovery = normalized_loss / (recovery_time + EPS)

            # Ai based on sums within event.
            gpp_neg = gpp_ts[t0 : t1 + 1]
            mcdi_seg = mcdi_ts[t0 : t1 + 1]
            gpp_sum = np.nansum(gpp_neg[gpp_neg < threshold])
            mcdi_sum = np.nansum(mcdi_seg[mcdi_seg < -1])
            Ai = gpp_sum / (mcdi_sum + EPS)

            Aprev = np.nan
            if len(prev_ai) > 0:
                Aprev = float(np.nanmean(prev_ai))

            adaptability = np.nan
            if np.isfinite(Aprev) and np.isfinite(Ai) and not np.isclose(Ai, 0):
                adaptability = Aprev / Ai

            out.at[idx, "normalized_loss"] = normalized_loss
            out.at[idx, "resistance_time"] = resistance_time
            out.at[idx, "recovery_time"] = recovery_time
            out.at[idx, "resistance_raw"] = resistance
            out.at[idx, "recovery_raw"] = recovery
            out.at[idx, "Ai"] = Ai
            out.at[idx, "Aprev"] = Aprev
            out.at[idx, "adaptability_raw"] = adaptability

            prev_ai.append(Ai)

    out["resistance"] = minmax_01(out["resistance_raw"])
    out["recovery"] = minmax_01(out["recovery_raw"])
    out["adaptability"] = minmax_01(out["adaptability_raw"])
    return out


def entropy_weights(df: pd.DataFrame, cols: Iterable[str]) -> Dict[str, float]:
    """Compute entropy weights for indicator columns."""
    X = df[list(cols)].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        return {c: np.nan for c in cols}

    # Ensure positive and normalized for probability.
    X = X - X.min() + EPS
    P = X.div(X.sum(axis=0), axis=1)
    n = len(P)
    e = -(P * np.log(P + EPS)).sum(axis=0) / np.log(n)
    d = 1 - e
    w = d / d.sum()
    return w.to_dict()


def compute_resilience(events_metrics: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute entropy weighted resilience index from three dimensions."""
    cols = ["resistance", "recovery", "adaptability"]
    weights = entropy_weights(events_metrics, cols)

    df = events_metrics.copy()
    df["resilience"] = (
        weights["resistance"] * df["resistance"]
        + weights["recovery"] * df["recovery"]
        + weights["adaptability"] * df["adaptability"]
    )
    return df, weights


def aggregate_to_maps(events_df: pd.DataFrame) -> xr.Dataset:
    """Aggregate event metrics to pixel-wise mean maps."""
    agg = (
        events_df.groupby(["lat", "lon"])[
            ["resistance", "recovery", "adaptability", "resilience"]
        ]
        .mean()
        .reset_index()
    )

    ds = agg.set_index(["lat", "lon"]).to_xarray()
    return ds


def ar1_1d(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation for one time series."""
    x0 = x[:-1]
    x1 = x[1:]
    mask = np.isfinite(x0) & np.isfinite(x1)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x0[mask], x1[mask])[0, 1])


def rolling_ar1(gpp_detrended: xr.DataArray, window: int = 60) -> xr.DataArray:
    """Compute rolling AR(1) along time for each pixel."""

    def _roll(ts: np.ndarray) -> np.ndarray:
        out = np.full(ts.shape[0], np.nan, dtype=float)
        for t in range(window, ts.shape[0] + 1):
            out[t - 1] = ar1_1d(ts[t - window : t])
        return out

    return xr.apply_ufunc(
        _roll,
        gpp_detrended,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
