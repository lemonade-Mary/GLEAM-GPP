"""End-to-end drought resilience pipeline.

Usage example
-------------
python run_pipeline.py \
  --nc_file "C:/Users/Administrator/outputs/Yunnan_EcoHydrology_Final.nc" \
  --outdir outputs \
  --resolution 0.05

The script is designed to be directly adaptable to your own NetCDF data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from drought_detection import detect_drought_events
from model import TARGETS, prepare_model_table, run_shap, train_xgb_cv
from preprocess import PreprocessConfig, preprocess_dataset
from resilience import aggregate_to_maps, compute_event_metrics, compute_resilience, rolling_ar1


def plot_metric_maps(ds_map: xr.Dataset, outdir: Path) -> List[str]:
    paths = []
    for var in ["resistance", "recovery", "adaptability", "resilience"]:
        if var not in ds_map:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ds_map[var].plot(ax=ax, cmap="RdYlBu_r")
        ax.set_title(f"Spatial {var.capitalize()}")
        p = outdir / f"map_{var}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(str(p))
    return paths


def plot_ecosystem_timeseries(events: pd.DataFrame, outdir: Path) -> str | None:
    if "eco_type" not in events.columns or "start_time" not in events.columns:
        return None

    ts = (
        events.assign(year=pd.to_datetime(events["start_time"]).dt.year)
        .groupby(["year", "eco_type"])[["resistance", "recovery", "adaptability", "resilience"]]
        .mean()
        .reset_index()
    )
    if ts.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for i, metric in enumerate(["resistance", "recovery", "adaptability", "resilience"]):
        ax = axes[i]
        for eco, df_eco in ts.groupby("eco_type"):
            ax.plot(df_eco["year"], df_eco[metric], label=str(eco))
        ax.set_title(metric.capitalize())
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    p = outdir / "timeseries_by_ecosystem.png"
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return str(p)


def attach_drought_features(events: pd.DataFrame, ds: xr.Dataset, drought_var: str = "spei_anom") -> pd.DataFrame:
    """Add drought frequency / severity / duration features for modeling."""
    events = events.copy()
    events["severity"] = np.nan
    events["frequency"] = np.nan

    grouped = events.groupby(["lat", "lon"], sort=False)
    for (lat, lon), idxs in grouped.groups.items():
        pixel_events = events.loc[idxs].sort_values("start_idx")
        yi = int(np.argmin(np.abs(ds.lat.values - lat)))
        xi = int(np.argmin(np.abs(ds.lon.values - lon)))
        mcdi = ds[drought_var][:, yi, xi].values

        n_prev = 0
        for idx, row in pixel_events.iterrows():
            t0, t1 = int(row.start_idx), int(row.end_idx)
            seg = mcdi[t0 : t1 + 1]
            severity = np.nansum(seg[seg < -1])
            events.at[idx, "severity"] = severity
            events.at[idx, "frequency"] = n_prev
            n_prev += 1

    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_file", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--gpp_var", type=str, default="gpp_anom")
    parser.add_argument("--drought_var", type=str, default="spei_anom")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.nc_file, chunks={"time": -1, "lat": 64, "lon": 64})

    cfg = PreprocessConfig(
        target_resolution=args.resolution,
        gpp_var=args.gpp_var,
        drought_var=args.drought_var,
    )
    ds_p = preprocess_dataset(ds, cfg)

    events = detect_drought_events(
        ds_p,
        drought_var=args.drought_var,
        gpp_detrended_var="gpp_detrended",
        gpp_neg_mask_var="gpp_neg_mask",
        min_duration=3,
        threshold=-1.0,
    )
    events.to_csv(outdir / "drought_events.csv", index=False)

    metrics = compute_event_metrics(events, ds_p, gpp_detrended_var="gpp_detrended", drought_var=args.drought_var)
    metrics, weights = compute_resilience(metrics)
    metrics = attach_drought_features(metrics, ds_p, drought_var=args.drought_var)
    metrics.to_csv(outdir / "event_metrics_resilience.csv", index=False)

    pd.Series(weights).to_csv(outdir / "entropy_weights.csv", header=["weight"])

    ds_map = aggregate_to_maps(metrics)
    ds_map.to_netcdf(outdir / "resilience_maps.nc")
    plot_metric_maps(ds_map, outdir)
    plot_ecosystem_timeseries(metrics, outdir)

    ar1_da = rolling_ar1(ds_p["gpp_detrended"], window=60)
    ar1_mean = ar1_da.mean(dim=["lat", "lon"], skipna=True).compute()
    ar1_mean.to_netcdf(outdir / "ar1_series.nc")

    fig, ax = plt.subplots(figsize=(9, 4))
    ar1_mean.plot(ax=ax, label="Mean AR(1)")
    ax.set_title("AR(1) validation series")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ar1_timeseries.png", dpi=220)
    plt.close(fig)

    # If dynamic predictors are not pre-joined at event level, sample from event start month.
    need_sample = [v for v in ["sm_anom", "t2m_anom", "rad_anom", "evi_anom", "gosif_anom", "spei_anom"] if v in ds_p]
    if need_sample:
        for v in need_sample:
            metrics[v] = np.nan
        for i, r in metrics.iterrows():
            yi = int(np.argmin(np.abs(ds_p.lat.values - r.lat)))
            xi = int(np.argmin(np.abs(ds_p.lon.values - r.lon)))
            ti = int(r.start_idx)
            for v in need_sample:
                metrics.at[i, v] = float(ds_p[v][ti, yi, xi].values)

    feature_candidates = [
        c
        for c in [
            "duration",
            "severity",
            "frequency",
            "sm_anom",
            "t2m_anom",
            "rad_anom",
            "evi_anom",
            "gosif_anom",
            "spei_anom",
            "dem",
            "awc",
            "cec",
        ]
        if c in metrics.columns
    ]

    model_table = prepare_model_table(metrics, dynamic_features=feature_candidates, static_df=None)

    model_scores = {}
    for target in TARGETS:
        if target not in model_table.columns:
            continue
        feature_cols = [c for c in feature_candidates if c in model_table.columns and c != target]
        if len(feature_cols) < 3:
            continue
        model, score, oof, pred_df = train_xgb_cv(model_table, target=target, feature_cols=feature_cols)
        model_scores[target] = score
        pred_df.to_csv(outdir / f"cv_pred_{target}.csv", index=False)
        shap_out = run_shap(model, model_table[feature_cols], outdir=str(outdir), target_name=target)
        print(f"[{target}] R2={score:.3f}, SHAP outputs: {shap_out}")

    pd.Series(model_scores).to_csv(outdir / "xgboost_cv_r2.csv", header=["r2"])


if __name__ == "__main__":
    main()
