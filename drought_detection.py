"""Drought event detection.

Rules (paper-consistent):
- Drought starts when MCDI <= -1.
- Drought ends when MCDI > -1.
- Event duration must be >= 3 months.
- Event must overlap with negative GPP anomalies (gpp < -0.5 SD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class DroughtEvent:
    lat: float
    lon: float
    start_idx: int
    end_idx: int
    min_idx: int
    duration: int
    min_gpp: float



def _find_events_1d(
    mcdi: np.ndarray,
    gpp_detrended: np.ndarray,
    neg_mask: np.ndarray,
    min_duration: int = 3,
    threshold: float = -1.0,
) -> List[tuple]:
    """Find valid drought events for one pixel."""
    is_drought = np.isfinite(mcdi) & (mcdi <= threshold)
    n = len(mcdi)
    events: List[tuple] = []

    i = 0
    while i < n:
        if not is_drought[i]:
            i += 1
            continue

        start = i
        while i < n and is_drought[i]:
            i += 1
        end = i - 1

        duration = end - start + 1
        if duration < min_duration:
            continue

        if not np.any(neg_mask[start : end + 1]):
            continue

        seg = gpp_detrended[start : end + 1]
        if not np.any(np.isfinite(seg)):
            continue

        rel_min = int(np.nanargmin(seg))
        min_idx = start + rel_min
        min_gpp = float(seg[rel_min])
        events.append((start, end, min_idx, duration, min_gpp))

    return events


def detect_drought_events(
    ds: xr.Dataset,
    drought_var: str = "spei_anom",
    gpp_detrended_var: str = "gpp_detrended",
    gpp_neg_mask_var: str = "gpp_neg_mask",
    min_duration: int = 3,
    threshold: float = -1.0,
) -> pd.DataFrame:
    """Detect drought events for every pixel and return tabular event list."""
    lat_vals = ds.lat.values
    lon_vals = ds.lon.values

    records = []
    for yi, lat in enumerate(lat_vals):
        for xi, lon in enumerate(lon_vals):
            mcdi = ds[drought_var][:, yi, xi].values
            gpp_det = ds[gpp_detrended_var][:, yi, xi].values
            neg_mask = ds[gpp_neg_mask_var][:, yi, xi].values
            events = _find_events_1d(
                mcdi=mcdi,
                gpp_detrended=gpp_det,
                neg_mask=neg_mask,
                min_duration=min_duration,
                threshold=threshold,
            )
            for (start, end, min_idx, duration, min_gpp) in events:
                records.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "min_idx": int(min_idx),
                        "duration": int(duration),
                        "min_gpp": float(min_gpp),
                        "start_time": pd.Timestamp(ds.time.values[start]),
                        "end_time": pd.Timestamp(ds.time.values[end]),
                    }
                )

    return pd.DataFrame.from_records(records)
