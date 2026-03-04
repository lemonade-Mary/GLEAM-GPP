import argparse
import os
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def compute_monthly_detrended_anomaly_df(series: pd.Series, value_name: str) -> pd.DataFrame:
    """计算当月距平与去趋势异常：后续所有分析统一使用 anomaly_detrended。"""
    df = pd.DataFrame({value_name: series}).copy()
    monthly_mean = df[value_name].groupby(df.index.month).transform("mean")
    df["anomaly"] = df[value_name] - monthly_mean
    valid = df["anomaly"].notna()
    df["anomaly_detrended"] = np.nan
    if valid.any():
        df.loc[valid, "anomaly_detrended"] = detrend(df.loc[valid, "anomaly"].to_numpy())
    return df


def compute_monthly_detrended_anomaly(series: pd.Series) -> pd.Series:
    return compute_monthly_detrended_anomaly_df(series, "value")["anomaly_detrended"]


def masked_mean_raster(path: Path, shapefile: gpd.GeoDataFrame) -> float:
    with rasterio.open(path) as src:
        out_image, _ = mask(src, [shapefile.geometry.union_all()], crop=True)
        arr = out_image[0].astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        return float(np.nanmean(arr))


def load_gpp_monthly_mean(gpp_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex, gpp_pattern: str):
    means = []
    for t in period:
        path = gpp_dir / gpp_pattern.format(year=t.year, month=t.month)
        means.append(masked_mean_raster(path, shapefile) if path.exists() else np.nan)
    return pd.Series(means, index=period, name="gpp")


def load_gpp_stack(gpp_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex, gpp_pattern: str):
    stack = []
    transform, crs, nodata = None, None, None
    for t in period:
        path = gpp_dir / gpp_pattern.format(year=t.year, month=t.month)
        if not path.exists():
            stack.append(None)
            continue
        with rasterio.open(path) as src:
            out_image, out_transform = mask(src, [shapefile.geometry.union_all()], crop=True)
            arr = out_image[0].astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[~np.isfinite(arr)] = np.nan
            if transform is None:
                transform, crs, nodata = out_transform, src.crs, src.nodata
            stack.append(arr)

    template = next((x for x in stack if x is not None), None)
    if template is None:
        raise ValueError("未读取到任何 GPP 栅格。")
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
    out = np.full((n_time, n_row, n_col), np.nan, dtype=float)
    for r in range(n_row):
        for c in range(n_col):
            ts = data_3d[:, r, c]
            if np.all(np.isnan(ts)):
                continue
            out[:, r, c] = compute_monthly_detrended_anomaly(pd.Series(ts, index=times)).to_numpy()
    return out


def timeseries_plot(sm_series: pd.Series, gpp_series: pd.Series, out_png: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(sm_series.index, sm_series.values, color="tab:blue", linewidth=1.4)
    axes[0].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("去趋势土壤水分异常 (m³/m³)")
    axes[0].set_title("云南省土壤水分去趋势异常时间序列")
    axes[0].grid(alpha=0.3)

    axes[1].plot(gpp_series.index, gpp_series.values, color="tab:green", linewidth=1.4)
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im1 = axes[0].imshow(sm_drought, cmap="RdBu_r")
    axes[0].scatter(sm_worst_idx[1], sm_worst_idx[0], c="black", s=45, label="SM最严重像元")
    axes[0].set_title("2009-2010 土壤水分去趋势异常均值")
    axes[0].set_xlabel("列")
    axes[0].set_ylabel("行")
    axes[0].legend(loc="lower right")
    cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cb1.set_label("m³/m³")

    im2 = axes[1].imshow(gpp_drought, cmap="RdBu_r")
    axes[1].scatter(sm_worst_idx[1], sm_worst_idx[0], c="black", s=45, label="SM最严重像元")
    axes[1].scatter(gpp_worst_idx[1], gpp_worst_idx[0], c="yellow", s=45, label="GPP最严重像元")
    axes[1].set_title("2009-2010 GPP 去趋势异常均值")
    axes[1].set_xlabel("列")
    axes[1].set_ylabel("行")
    axes[1].legend(loc="lower right")
    cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cb2.set_label("g C m⁻² month⁻¹")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    sm_x, sm_y = rasterio.transform.xy(transform, sm_worst_idx[0], sm_worst_idx[1])
    gpp_x, gpp_y = rasterio.transform.xy(transform, gpp_worst_idx[0], gpp_worst_idx[1])

    return {
        "sm_worst_pixel": sm_worst_idx,
        "sm_worst_lonlat": (sm_x, sm_y),
        "sm_worst_value": float(sm_drought[sm_worst_idx]),
        "gpp_at_sm_worst": float(gpp_drought[sm_worst_idx]),
        "gpp_worst_pixel": gpp_worst_idx,
        "gpp_worst_lonlat": (gpp_x, gpp_y),
        "gpp_worst_value": float(gpp_drought[gpp_worst_idx]),
        "same_pixel": sm_worst_idx == gpp_worst_idx,
    }


def compute_recovery_months(gpp_anom, times, event_years=(2009, 2010), max_followup=24):
    event_idx = np.where((times.year >= event_years[0]) & (times.year <= event_years[1]))[0]
    n_time, n_row, n_col = gpp_anom.shape
    recovery = np.full((n_row, n_col), np.nan, dtype=float)
    for r in range(n_row):
        for c in range(n_col):
            ts = gpp_anom[:, r, c]
            event_ts = ts[event_idx]
            if np.all(np.isnan(event_ts)):
                continue
            min_idx = event_idx[np.nanargmin(event_ts)]
            follow = ts[min_idx : min(n_time, min_idx + max_followup + 1)]
            hit = np.where(follow >= 0)[0]
            recovery[r, c] = float(hit[0]) if len(hit) > 0 else np.nan
    return recovery


def landcover_recovery_plot(recovery_months, landcover_tif: Path, gpp_transform, gpp_crs, out_png: Path, out_csv: Path):
    with rasterio.open(landcover_tif) as lc_ds:
        lc = lc_ds.read(1).astype(float)
        lc_meta = lc_ds.meta.copy()
        if lc_ds.nodata is not None:
            lc[lc == lc_ds.nodata] = np.nan

    height, width = recovery_months.shape
    if lc.shape != (height, width):
        dst_lc = np.empty((height, width), dtype=float)
        reproject(
            source=lc,
            destination=dst_lc,
            src_transform=lc_meta["transform"],
            src_crs=lc_meta["crs"],
            dst_transform=Affine.from_gdal(*gpp_transform.to_gdal()),
            dst_crs=gpp_crs,
            resampling=Resampling.nearest,
        )
        lc = dst_lc

    valid = np.isfinite(lc) & np.isfinite(recovery_months)
    df = pd.DataFrame({"LandCover": lc[valid].astype(int), "RecoveryMonths": recovery_months[valid]})

    igbp = {
        0: "Water", 1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest", 3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest", 5: "Mixed Forests", 6: "Closed Shrublands", 7: "Open Shrublands",
        8: "Woody Savannas", 9: "Savannas", 10: "Grasslands", 11: "Permanent Wetlands", 12: "Croplands",
        13: "Urban and Built-up", 14: "Cropland/Natural Veg. Mosaic", 15: "Snow and Ice", 16: "Barren or Sparsely Vegetated",
    }
    df["LandCoverName"] = df["LandCover"].map(igbp)
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

    summary = df.groupby("LandCoverName")["RecoveryMonths"].agg(["mean", "median", "std", "count"]).reset_index()
    summary.sort_values("mean", ascending=False).to_csv(out_csv, index=False, encoding="utf-8-sig")


def load_modis_ndvi_evi_series(modis_dir: Path, period: pd.DatetimeIndex):
    try:
        from osgeo import gdal
    except Exception as e:  # pragma: no cover
        raise RuntimeError("读取 MODIS HDF 需要 GDAL（from osgeo import gdal）。") from e

    ndvi_data, evi_data = [], []
    cwd = os.getcwd()
    os.chdir(modis_dir)
    try:
        for t in period:
            filename = f"MOD13C2.A{t.year:04d}.M{t.month:02d}.hdf"
            if not Path(filename).exists():
                continue
            ndvi_ds = gdal.Open(f'HDF4_EOS:EOS_GRID:"{filename}":MOD_Grid_monthly_CMG_VI:CMG 0.05 Deg Monthly NDVI')
            evi_ds = gdal.Open(f'HDF4_EOS:EOS_GRID:"{filename}":MOD_Grid_monthly_CMG_VI:CMG 0.05 Deg Monthly EVI')
            if ndvi_ds is None or evi_ds is None:
                continue
            ndvi = ndvi_ds.ReadAsArray().astype(float)
            evi = evi_ds.ReadAsArray().astype(float)
            ndvi[ndvi == -3000] = np.nan
            evi[evi == -3000] = np.nan
            ndvi *= 0.0001
            evi *= 0.0001
            ndvi_data.append([t, float(np.nanmean(ndvi))])
            evi_data.append([t, float(np.nanmean(evi))])
    finally:
        os.chdir(cwd)

    ndvi_df = pd.DataFrame(ndvi_data, columns=["time", "ndvi"]).set_index("time") if ndvi_data else pd.DataFrame(columns=["ndvi"])
    evi_df = pd.DataFrame(evi_data, columns=["time", "evi"]).set_index("time") if evi_data else pd.DataFrame(columns=["evi"])
    return ndvi_df, evi_df


def load_gosif_series(gosif_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex, gosif_pattern: str):
    data = []
    for t in period:
        path = gosif_dir / gosif_pattern.format(year=t.year, month=t.month)
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            out_img, _ = mask(src, [shapefile.geometry.union_all()], crop=True)
            arr = out_img[0].astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[~np.isfinite(arr)] = np.nan
            data.append([t, float(np.nanmean(arr))])
    return pd.DataFrame(data, columns=["time", "gosif"]).set_index("time") if data else pd.DataFrame(columns=["gosif"])


def plot_multisource_sync(merged: pd.DataFrame, out_corr_png: Path, out_series_png: Path):
    corr = merged.corr(method="pearson")
    corr.to_csv(out_corr_png.with_suffix(".csv"), encoding="utf-8-sig")

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f")
    plt.title("多源去趋势异常相关矩阵（同步性）")
    plt.tight_layout()
    plt.savefig(out_corr_png, dpi=300)
    plt.close()

    plt.figure(figsize=(13, 6))
    for col in merged.columns:
        s = merged[col]
        if s.notna().sum() < 3:
            continue
        z = (s - s.mean()) / s.std()
        plt.plot(merged.index, z, lw=1, label=col)
    plt.axhline(0, color="k", ls="--", lw=0.8)
    plt.xlabel("时间")
    plt.ylabel("标准化去趋势异常 (z-score, 无量纲)")
    plt.title("云南省干旱与植被异常同步性")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_series_png, dpi=300)
    plt.close()


def plot_spei_subplots(spei_dir: Path, min_lon, max_lon, min_lat, max_lat, out_png: Path):
    def _plot(ax, filename, title):
        ds = xr.open_dataset(filename)
        lat_values = ds["lat"].values
        lat_slice = slice(min_lat, max_lat) if lat_values[0] < lat_values[-1] else slice(max_lat, min_lat)
        spei_data = ds["spei"].sel(lat=lat_slice, lon=slice(min_lon, max_lon))
        yunnan = spei_data.sel(time=slice("2000-01-01", "2022-12-31")).mean(dim=["lat", "lon"])
        t = pd.to_datetime(yunnan.time.values)
        ax.plot(t, yunnan.values, marker="o", markersize=1, color="#6b5458", linewidth=0.7)
        for i, tt in enumerate(t):
            if tt.month in [3, 4, 5]:
                ax.scatter(tt, yunnan.values[i], color="#a72126", zorder=5, s=12)
        ax.axhline(y=-0.5, color="r", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel(title)

    fig, axs = plt.subplots(4, 3, figsize=(18, 12))
    for i, ax in enumerate(axs.flat):
        m = i + 1
        filename = spei_dir / f"spei{m:02d}.nc"
        if filename.exists():
            _plot(ax, str(filename), f"SPEI{m:02d}")
        else:
            ax.set_title(f"SPEI{m:02d} missing")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def load_monthly_spei_series(spei_dir: Path, min_lon, max_lon, min_lat, max_lat):
    out = []
    for m in range(1, 13):
        fn = spei_dir / f"spei{m:02d}.nc"
        if not fn.exists():
            continue
        ds = xr.open_dataset(fn)
        lat_values = ds["lat"].values
        lat_slice = slice(min_lat, max_lat) if lat_values[0] < lat_values[-1] else slice(max_lat, min_lat)
        s = ds["spei"].sel(lat=lat_slice, lon=slice(min_lon, max_lon)).mean(dim=["lat", "lon"]).to_pandas()
        s.index = pd.to_datetime(s.index)
        out.append(s[s.index.month == m])
    if len(out) == 0:
        return pd.Series(dtype=float)
    spei = pd.concat(out).sort_index()
    spei.name = "spei"
    return spei


def mark_consecutive_drought(spei: pd.Series, threshold=-0.5, min_len=2):
    flag = (spei < threshold).astype(int)
    grp = (flag != flag.shift(1)).cumsum()
    runlen = flag.groupby(grp).transform("sum")
    return ((flag == 1) & (runlen >= min_len)).astype(int)


def ml_drought_impact(merged: pd.DataFrame, ml_start: str, ml_end: str, out_prefix: Path):
    """机器学习分析 2009-2015 连续小干旱对 GPP 异常的影响。"""
    ml_df = merged.loc[ml_start:ml_end].copy()
    ml_df["drought_small"] = (ml_df["SPEI_anom_dt"] < -0.5).astype(float)
    ml_df["drought_consecutive"] = mark_consecutive_drought(ml_df["SPEI_anom_dt"])

    # 构造1~3个月滞后特征，考虑干旱影响植被/GPP的延迟响应。
    for lag in [1, 2, 3]:
        for col in ["SM_anom_dt", "NDVI_anom_dt", "EVI_anom_dt", "GOSIF_anom_dt", "SPEI_anom_dt"]:
            if col in ml_df.columns:
                ml_df[f"{col}_lag{lag}"] = ml_df[col].shift(lag)

    y = ml_df["GPP_anom_dt"]
    X = ml_df.drop(columns=["GPP_anom_dt"])
    X = X.dropna(axis=1, how="all")

    # 如果有效样本过少，直接退出并返回空结果。
    valid_rows = y.notna() & X.notna().any(axis=1)
    y = y.loc[valid_rows]
    X = X.loc[valid_rows]
    if len(y) < 24:
        pd.DataFrame().to_csv(out_prefix.with_name(out_prefix.name + "_ml_models.csv"), index=False)
        return

    tscv = TimeSeriesSplit(n_splits=5)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(C=1.0, epsilon=0.1),
    }

    results = []
    for name, model in models.items():
        pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]
        )
        neg_mse_scores = cross_val_score(pipe, X, y, cv=tscv, scoring="neg_mean_squared_error")
        rmse = float(np.mean(np.sqrt(-neg_mse_scores)))
        results.append([name, rmse])

    result_df = pd.DataFrame(results, columns=["model", "cv_rmse"]).sort_values("cv_rmse")
    result_df.to_csv(out_prefix.with_name(out_prefix.name + "_ml_models.csv"), index=False, encoding="utf-8-sig")

    best_name = result_df.iloc[0]["model"]
    best_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", models[best_name]),
        ]
    )
    best_pipe.fit(X, y)
    pred = best_pipe.predict(X)

    metrics_df = pd.DataFrame(
        [{"best_model": best_name, "train_r2": float(r2_score(y, pred)), "train_rmse": float(np.sqrt(mean_squared_error(y, pred)))}]
    )
    metrics_df.to_csv(out_prefix.with_name(out_prefix.name + "_ml_metrics.csv"), index=False, encoding="utf-8-sig")

    impact = ml_df[["GPP_anom_dt", "drought_consecutive"]].dropna()
    stats = impact.groupby("drought_consecutive")["GPP_anom_dt"].agg(["mean", "median", "std", "count"])
    stats.to_csv(out_prefix.with_name(out_prefix.name + "_drought_group_stats.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=impact, x="drought_consecutive", y="GPP_anom_dt")
    plt.xlabel("是否连续小干旱（SPEI<-0.5 且连续≥2个月）")
    plt.ylabel("GPP 去趋势异常 (g C m⁻² month⁻¹)")
    plt.title("连续小干旱对 GPP 异常影响（2009-2015）")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_drought_impact_boxplot.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="云南省干旱-植被多源联合分析（整合版）")
    parser.add_argument("--shapefile", required=True, help="云南省边界shp路径")
    parser.add_argument("--gpp-dir", required=True, help="FluxSat月尺度GPP目录")
    parser.add_argument("--sm-nc", required=True, help="GLEAM土壤水分nc路径")
    parser.add_argument("--landcover", required=True, help="IGBP土地利用tif路径")

    parser.add_argument("--modis-dir", required=True, help="MOD13C2 HDF目录（NDVI/EVI）")
    parser.add_argument("--gosif-dir", required=True, help="GOSIF月尺度tif目录")
    parser.add_argument("--spei-dir", required=True, help="SPEI目录（spei01.nc...spei12.nc）")

    parser.add_argument("--gpp-pattern", default="{year:04d}_{month:02d}_FluxSat.tif", help="GPP文件名模板")
    parser.add_argument("--gosif-pattern", default="GOSIF_{year:04d}.M{month:02d}.tif", help="GOSIF文件名模板")
    parser.add_argument("--sm-var", default="SMroot", help="土壤水分变量名")

    parser.add_argument("--start", default="2000-03-01", help="开始时间")
    parser.add_argument("--end", default="2022-05-01", help="结束时间")
    parser.add_argument("--ml-start", default="2009-01-01", help="机器学习开始时间")
    parser.add_argument("--ml-end", default="2015-12-01", help="机器学习结束时间")
    parser.add_argument("--outdir", default="outputs", help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shapefile = gpd.read_file(args.shapefile)
    period = month_range(args.start, args.end)
    min_lon, min_lat, max_lon, max_lat = shapefile.total_bounds

    # 1) 省域 SM/GPP 去趋势异常时间序列
    gpp_series = load_gpp_monthly_mean(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)
    gpp_df = compute_monthly_detrended_anomaly_df(gpp_series, "gpp")

    sm = load_sm_stack(Path(args.sm_nc), args.sm_var, shapefile, period)
    sm_series = sm.mean(dim=["lat", "lon"], skipna=True).to_pandas()
    sm_series.index = pd.to_datetime(sm_series.index)
    sm_df = compute_monthly_detrended_anomaly_df(sm_series, "soil_moisture")

    timeseries_plot(sm_df["anomaly_detrended"], gpp_df["anomaly_detrended"], outdir / "01_timeseries_sm_gpp.png")

    # 2) 2009-2010 严重干旱区域识别与 GPP 异常对比
    gpp_stack, gpp_transform, gpp_crs, _ = load_gpp_stack(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)
    gpp_anom_stack = compute_pixelwise_detrended_anomaly(gpp_stack, period)

    sm_arr = sm.to_numpy()
    sm_anom_stack = compute_pixelwise_detrended_anomaly(sm_arr, period)

    if sm_anom_stack.shape[1:] != gpp_anom_stack.shape[1:]:
        dst = np.empty_like(gpp_anom_stack)
        src_transform = Affine.translation(float(sm.lon.min()), float(sm.lat.max())) * Affine.scale(
            float(sm.lon[1] - sm.lon[0]), -abs(float(sm.lat[1] - sm.lat[0]))
        )
        for i in range(sm_anom_stack.shape[0]):
            reproject(
                source=sm_anom_stack[i],
                destination=dst[i],
                src_transform=src_transform,
                src_crs="EPSG:4326",
                dst_transform=gpp_transform,
                dst_crs=gpp_crs,
                resampling=Resampling.bilinear,
            )
        sm_anom_stack = dst

    drought_report = drought_region_analysis(
        sm_anom_stack, gpp_anom_stack, period, gpp_transform, outdir / "02_drought_region_sm_vs_gpp_2009_2010.png"
    )
    pd.DataFrame([drought_report]).to_csv(outdir / "02_drought_region_report.csv", index=False, encoding="utf-8-sig")

    # 3) 土地利用类型恢复分析
    recovery_months = compute_recovery_months(gpp_anom_stack, period, event_years=(2009, 2010), max_followup=24)
    landcover_recovery_plot(
        recovery_months,
        Path(args.landcover),
        gpp_transform,
        gpp_crs,
        outdir / "03_landcover_recovery_boxplot.png",
        outdir / "03_landcover_recovery_summary.csv",
    )

    # 4) NDVI/EVI、GOSIF 扩展并做同步性分析
    ndvi_raw, evi_raw = load_modis_ndvi_evi_series(Path(args.modis_dir), period)
    gosif_raw = load_gosif_series(Path(args.gosif_dir), shapefile, period, args.gosif_pattern)

    ndvi_df = compute_monthly_detrended_anomaly_df(ndvi_raw["ndvi"], "ndvi") if not ndvi_raw.empty else pd.DataFrame(index=period)
    evi_df = compute_monthly_detrended_anomaly_df(evi_raw["evi"], "evi") if not evi_raw.empty else pd.DataFrame(index=period)
    gosif_df = (
        compute_monthly_detrended_anomaly_df(gosif_raw["gosif"], "gosif") if not gosif_raw.empty else pd.DataFrame(index=period)
    )

    merged = pd.DataFrame(index=period)
    merged["SM_anom_dt"] = sm_df["anomaly_detrended"]
    merged["GPP_anom_dt"] = gpp_df["anomaly_detrended"]
    merged["NDVI_anom_dt"] = ndvi_df["anomaly_detrended"].reindex(period) if "anomaly_detrended" in ndvi_df else np.nan
    merged["EVI_anom_dt"] = evi_df["anomaly_detrended"].reindex(period) if "anomaly_detrended" in evi_df else np.nan
    merged["GOSIF_anom_dt"] = gosif_df["anomaly_detrended"].reindex(period) if "anomaly_detrended" in gosif_df else np.nan

    # 5) SPEI：子图 + 月序列融合
    plot_spei_subplots(Path(args.spei_dir), min_lon, max_lon, min_lat, max_lat, outdir / "04_spei_12_subplots.png")
    spei_series = load_monthly_spei_series(Path(args.spei_dir), min_lon, max_lon, min_lat, max_lat)
    if len(spei_series) > 0:
        spei_df = compute_monthly_detrended_anomaly_df(spei_series, "spei")
        merged["SPEI_anom_dt"] = spei_df["anomaly_detrended"].reindex(period)
    else:
        merged["SPEI_anom_dt"] = np.nan

    # 6) 多源同步性图件
    plot_multisource_sync(merged, outdir / "05_sync_corr_heatmap.png", outdir / "05_sync_zscore_timeseries.png")
    merged.to_csv(outdir / "05_merged_multisource_anomaly_timeseries.csv", encoding="utf-8-sig")

    # 7) 机器学习分析：2009-2015 连续小干旱影响
    ml_drought_impact(merged, args.ml_start, args.ml_end, outdir / "06")

    print("分析完成，输出文件：")
    for p in sorted(outdir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
