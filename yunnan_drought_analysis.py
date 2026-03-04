import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import xarray as xr
from affine import Affine
from rasterio.mask import mask
from rasterio.warp import Resampling, reproject
from scipy.signal import detrend


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def compute_monthly_detrended_anomaly(series: pd.Series) -> pd.Series:
    monthly_mean = series.groupby(series.index.month).transform("mean")
    anomaly = series - monthly_mean
    valid = anomaly.notna()
    detrended = pd.Series(np.nan, index=series.index, dtype=float)
    detrended.loc[valid] = detrend(anomaly.loc[valid].to_numpy())
    return detrended


def load_gpp_monthly_mean(
    gpp_dir: Path,
    shapefile: gpd.GeoDataFrame,
    period: pd.DatetimeIndex,
    gpp_pattern: str,
):
    means = []
    shapes = [shapefile.geometry.unary_union]

    for t in period:
        filename = gpp_dir / gpp_pattern.format(year=t.year, month=t.month)
        if not filename.exists():
            means.append(np.nan)
            continue

        with rasterio.open(filename) as src:
            out_image, _ = mask(src, shapes, crop=True)
            band = out_image[0].astype(float)
            band[~np.isfinite(band)] = np.nan
            means.append(np.nanmean(band))

    return pd.Series(means, index=period, name="gpp")


def load_gpp_stack(
    gpp_dir: Path,
    shapefile: gpd.GeoDataFrame,
    period: pd.DatetimeIndex,
    gpp_pattern: str,
):
    stack = []
    transform = None
    crs = None
    nodata = None

    for t in period:
        filename = gpp_dir / gpp_pattern.format(year=t.year, month=t.month)
        if not filename.exists():
            stack.append(None)
            continue

        with rasterio.open(filename) as src:
            out_image, out_transform = mask(src, [shapefile.geometry.unary_union], crop=True)
            data = out_image[0].astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data[~np.isfinite(data)] = np.nan

            if transform is None:
                transform = out_transform
                crs = src.crs
                nodata = src.nodata

            stack.append(data)

    template = next((arr for arr in stack if arr is not None), None)
    if template is None:
        raise ValueError("未读取到任何 GPP 栅格，请检查路径与命名格式。")

    for i, arr in enumerate(stack):
        if arr is None:
            stack[i] = np.full_like(template, np.nan, dtype=float)

    return np.stack(stack), transform, crs, nodata


def load_sm_stack(nc_path: Path, var_name: str, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex):
    ds = xr.open_dataset(nc_path)
    sm = ds[var_name].sel(time=slice(str(period.min().date()), str(period.max().date())))

    min_lon, min_lat, max_lon, max_lat = shapefile.total_bounds
    sm = sm.sel(lon=slice(min_lon, max_lon), lat=slice(max_lat, min_lat))

    return sm


def compute_pixelwise_detrended_anomaly(data_3d: np.ndarray, times: pd.DatetimeIndex):
    n_time, n_row, n_col = data_3d.shape
    monthly = np.array([t.month for t in times])
    out = np.full_like(data_3d, np.nan, dtype=float)

    for r in range(n_row):
        for c in range(n_col):
            ts = data_3d[:, r, c]
            if np.all(np.isnan(ts)):
                continue
            s = pd.Series(ts, index=times)
            out[:, r, c] = compute_monthly_detrended_anomaly(s).to_numpy()

    return out


def timeseries_plot(sm_series: pd.Series, gpp_series: pd.Series, out_png: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(sm_series.index, sm_series.values, color="tab:blue", linewidth=1.6)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("去趋势土壤水分异常 (m³/m³)")
    axes[0].set_title("云南省土壤水分去趋势异常时间序列")
    axes[0].grid(alpha=0.3)

    axes[1].plot(gpp_series.index, gpp_series.values, color="tab:green", linewidth=1.6)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("去趋势 GPP 异常 (g C m⁻² month⁻¹)")
    axes[1].set_xlabel("时间")
    axes[1].set_title("云南省 GPP 去趋势异常时间序列")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def drought_region_analysis(sm_anom, gpp_anom, times, transform, out_png: Path):
    drought_mask = (times.year >= 2009) & (times.year <= 2010)

    sm_drought = np.nanmean(sm_anom[drought_mask], axis=0)
    gpp_drought = np.nanmean(gpp_anom[drought_mask], axis=0)

    sm_worst_idx = np.unravel_index(np.nanargmin(sm_drought), sm_drought.shape)
    gpp_worst_idx = np.unravel_index(np.nanargmin(gpp_drought), gpp_drought.shape)

    sm_worst_val = sm_drought[sm_worst_idx]
    gpp_at_sm_worst = gpp_drought[sm_worst_idx]
    gpp_worst_val = gpp_drought[gpp_worst_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im1 = axes[0].imshow(sm_drought, cmap="RdBu_r")
    axes[0].scatter(sm_worst_idx[1], sm_worst_idx[0], c="black", s=45, label="SM最严重像元")
    axes[0].set_title("2009-2010 土壤水分去趋势异常均值")
    axes[0].set_xlabel("列")
    axes[0].set_ylabel("行")
    axes[0].legend(loc="lower right")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("m³/m³")

    im2 = axes[1].imshow(gpp_drought, cmap="RdBu_r")
    axes[1].scatter(sm_worst_idx[1], sm_worst_idx[0], c="black", s=45, label="SM最严重像元")
    axes[1].scatter(gpp_worst_idx[1], gpp_worst_idx[0], c="yellow", s=45, label="GPP最严重像元")
    axes[1].set_title("2009-2010 GPP去趋势异常均值")
    axes[1].set_xlabel("列")
    axes[1].set_ylabel("行")
    axes[1].legend(loc="lower right")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("g C m⁻² month⁻¹")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    sm_x, sm_y = rasterio.transform.xy(transform, sm_worst_idx[0], sm_worst_idx[1])
    gpp_x, gpp_y = rasterio.transform.xy(transform, gpp_worst_idx[0], gpp_worst_idx[1])

    return {
        "sm_worst_pixel": sm_worst_idx,
        "sm_worst_lonlat": (sm_x, sm_y),
        "sm_worst_value": float(sm_worst_val),
        "gpp_at_sm_worst": float(gpp_at_sm_worst),
        "gpp_worst_pixel": gpp_worst_idx,
        "gpp_worst_lonlat": (gpp_x, gpp_y),
        "gpp_worst_value": float(gpp_worst_val),
        "same_pixel": sm_worst_idx == gpp_worst_idx,
    }


def compute_recovery_months(gpp_anom, times, event_years=(2009, 2010), max_followup=24):
    event_mask = (times.year >= event_years[0]) & (times.year <= event_years[1])
    event_idx = np.where(event_mask)[0]

    n_time, n_row, n_col = gpp_anom.shape
    recovery = np.full((n_row, n_col), np.nan, dtype=float)

    for r in range(n_row):
        for c in range(n_col):
            ts = gpp_anom[:, r, c]
            event_ts = ts[event_idx]
            if np.all(np.isnan(event_ts)):
                continue

            local_min_pos = np.nanargmin(event_ts)
            min_idx = event_idx[local_min_pos]

            end_idx = min(n_time, min_idx + max_followup + 1)
            follow = ts[min_idx:end_idx]
            hit = np.where(follow >= 0)[0]
            if len(hit) == 0:
                recovery[r, c] = np.nan
            else:
                recovery[r, c] = float(hit[0])

    return recovery


def landcover_recovery_plot(
    recovery_months,
    landcover_tif: Path,
    gpp_transform,
    gpp_crs,
    out_png: Path,
    out_csv: Path,
):
    with rasterio.open(landcover_tif) as lc_ds:
        lc = lc_ds.read(1).astype(float)
        lc_meta = lc_ds.meta.copy()
        if lc_ds.nodata is not None:
            lc[lc == lc_ds.nodata] = np.nan

    height, width = recovery_months.shape

    if lc.shape != (height, width):
        dst_lc = np.empty((height, width), dtype=float)
        dst_transform = Affine.from_gdal(*gpp_transform.to_gdal())
        reproject(
            source=lc,
            destination=dst_lc,
            src_transform=lc_meta["transform"],
            src_crs=lc_meta["crs"],
            dst_transform=dst_transform,
            dst_crs=gpp_crs,
            resampling=Resampling.nearest,
        )
        lc = dst_lc

    valid_mask = np.isfinite(lc) & np.isfinite(recovery_months)
    df = pd.DataFrame(
        {
            "LandCover": lc[valid_mask].astype(int),
            "RecoveryMonths": recovery_months[valid_mask],
        }
    )

    igbp_classes = {
        0: "Water",
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forests",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Veg. Mosaic",
        15: "Snow and Ice",
        16: "Barren or Sparsely Vegetated",
    }
    df["LandCoverName"] = df["LandCover"].map(igbp_classes)
    df = df.dropna(subset=["LandCoverName"])

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=df, x="LandCoverName", y="RecoveryMonths", palette="Spectral", showfliers=False)
    plt.xticks(rotation=75, ha="right")
    plt.xlabel("土地利用类型（IGBP）")
    plt.ylabel("植被恢复时间（月）")
    plt.title("不同土地利用类型下的植被恢复时间分布（云南省）")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    summary = (
        df.groupby("LandCoverName")["RecoveryMonths"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="云南省 2000-2022 土壤水分与GPP干旱分析")
    parser.add_argument("--shapefile", required=True, help="云南省边界shp路径")
    parser.add_argument("--gpp-dir", required=True, help="FluxSat月尺度GPP目录")
    parser.add_argument("--gpp-pattern", default="{year:04d}_{month:02d}_FluxSat.tif", help="GPP文件名模板")
    parser.add_argument("--sm-nc", required=True, help="GLEAM土壤水分nc路径")
    parser.add_argument("--sm-var", default="SMroot", help="土壤水分变量名")
    parser.add_argument("--landcover", required=True, help="IGBP土地利用tif路径")
    parser.add_argument("--outdir", default="outputs", help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shapefile = gpd.read_file(args.shapefile)
    period = month_range("2000-03-01", "2022-05-01")

    gpp_series = load_gpp_monthly_mean(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)
    gpp_series_anom = compute_monthly_detrended_anomaly(gpp_series)

    sm = load_sm_stack(Path(args.sm_nc), args.sm_var, shapefile, period)
    sm_mean_series = sm.mean(dim=["lat", "lon"], skipna=True).to_pandas()
    sm_mean_series.index = pd.to_datetime(sm_mean_series.index)
    sm_series_anom = compute_monthly_detrended_anomaly(sm_mean_series)

    timeseries_plot(sm_series_anom, gpp_series_anom, outdir / "01_timeseries_sm_gpp.png")

    gpp_stack, gpp_transform, gpp_crs, _ = load_gpp_stack(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)
    gpp_anom_stack = compute_pixelwise_detrended_anomaly(gpp_stack, period)

    sm_arr = sm.to_numpy()
    sm_anom_stack = compute_pixelwise_detrended_anomaly(sm_arr, period)

    if sm_anom_stack.shape[1:] != gpp_anom_stack.shape[1:]:
        dst = np.empty_like(gpp_anom_stack)
        for i in range(sm_anom_stack.shape[0]):
            reproject(
                source=sm_anom_stack[i],
                destination=dst[i],
                src_transform=Affine.translation(float(sm.lon.min()), float(sm.lat.max()))
                * Affine.scale(float(sm.lon[1] - sm.lon[0]), -abs(float(sm.lat[1] - sm.lat[0]))),
                src_crs="EPSG:4326",
                dst_transform=gpp_transform,
                dst_crs=gpp_crs,
                resampling=Resampling.bilinear,
            )
        sm_anom_stack = dst

    report = drought_region_analysis(
        sm_anom_stack,
        gpp_anom_stack,
        period,
        gpp_transform,
        outdir / "02_drought_region_sm_vs_gpp_2009_2010.png",
    )

    pd.DataFrame([report]).to_csv(outdir / "02_drought_region_report.csv", index=False, encoding="utf-8-sig")

    recovery_months = compute_recovery_months(gpp_anom_stack, period, event_years=(2009, 2010), max_followup=24)

    landcover_recovery_plot(
        recovery_months,
        Path(args.landcover),
        gpp_transform,
        gpp_crs,
        outdir / "03_landcover_recovery_boxplot.png",
        outdir / "03_landcover_recovery_summary.csv",
    )

    print("分析完成，输出文件：")
    for p in sorted(outdir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
