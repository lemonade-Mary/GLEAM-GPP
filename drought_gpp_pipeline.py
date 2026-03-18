import argparse
from pathlib import Path
import importlib

import geopandas as gpd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.mask import mask
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    """✅可直接使用：按月初生成时间序列。"""
    return pd.date_range(start=start, end=end, freq="MS")


def reproject_resample_to_wgs84(input_tif, output_tif, resolution=0.05, resampling_method="bilinear"):
    """✅可直接使用：将栅格统一到 WGS84 + 0.05°。"""
    resampling_dict = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}
    with rasterio.open(input_tif) as src:
        left, bottom, right, top = src.bounds
        dst_transform = from_origin(left, top, resolution, resolution)
        dst_width = int((right - left) / resolution)
        dst_height = int((top - bottom) / resolution)
        dst_array = np.empty((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=resampling_dict[resampling_method],
        )
        profile = src.profile
        profile.update(crs="EPSG:4326", transform=dst_transform, width=dst_width, height=dst_height, dtype="float32")
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(dst_array, 1)


def compute_monthly_detrended_anomaly_df(series, varname):
    """✅可直接使用：月气候态距平 + 线性去趋势。"""
    df = pd.DataFrame(series)
    df.columns = [varname]
    df.index = pd.to_datetime(df.index)
    climatology = df.groupby(df.index.month).transform("mean")
    df["anomaly"] = df[varname] - climatology[varname]
    x = np.arange(len(df))
    mask_valid = ~df["anomaly"].isna()
    coef = np.polyfit(x[mask_valid], df["anomaly"][mask_valid], 1)
    trend = np.polyval(coef, x)
    df["anomaly_detrended"] = df["anomaly"] - trend
    return df


def compute_monthly_detrended_anomaly(series: pd.Series) -> pd.Series:
    return compute_monthly_detrended_anomaly_df(series, "value")["anomaly_detrended"]


def masked_mean_raster(path: Path, shapefile: gpd.GeoDataFrame) -> float:
    """❗重写：使用真实矢量掩膜而不是 bbox-window。"""
    with rasterio.open(path) as src:
        shp = shapefile.to_crs(src.crs)
        out_img, _ = mask(src, [shp.geometry.unary_union], crop=True)
        arr = out_img[0].astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        return float(np.nanmean(arr))


def load_gpp_stack(gpp_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex, gpp_pattern: str):
    """✅可直接使用：逐月读取并裁剪 GPP 栅格。"""
    stack = []
    transform, crs, nodata = None, None, None
    for t in period:
        path = gpp_dir / gpp_pattern.format(year=t.year, month=t.month)
        if not path.exists():
            stack.append(None)
            continue
        with rasterio.open(path) as src:
            out_image, out_transform = mask(src, [shapefile.to_crs(src.crs).geometry.unary_union], crop=True)
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


def load_sm_stack(sm_nc, sm_var, shapefile, period):
    """❗重写：返回 numpy 3D（time,row,col）方便后续像元计算。"""
    import rioxarray  # noqa: F401

    ds = xr.open_dataset(sm_nc)
    da = ds[sm_var]
    da["time"] = da.indexes["time"].to_period("M").to_timestamp()
    da = da.sel(time=slice(period[0], period[-1]))
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    shp = shapefile.to_crs("EPSG:4326")
    da = da.rio.clip(shp.geometry, shp.crs, drop=True)
    return da.transpose("time", "lat", "lon").values


def compute_pixelwise_detrended_anomaly(data_3d: np.ndarray, times: pd.DatetimeIndex):
    """✅可直接使用：逐像元去趋势异常。"""
    n_time, n_row, n_col = data_3d.shape
    out = np.full((n_time, n_row, n_col), np.nan, dtype=float)
    for r in range(n_row):
        for c in range(n_col):
            ts = data_3d[:, r, c]
            if np.all(np.isnan(ts)):
                continue
            out[:, r, c] = compute_monthly_detrended_anomaly(pd.Series(ts, index=times)).to_numpy()
    return out


def compute_sm_era5_xr(sm1, sm2, sm3, sm4):
    """✅可直接使用：ERA5 四层深度加权。"""
    weights = [0.07, 0.21, 0.72, 1.89]
    return (sm1 * weights[0] + sm2 * weights[1] + sm3 * weights[2] + sm4 * weights[3]) / sum(weights)


def load_sm_era5_stack(sm_nc, sm_vars, shapefile_gdf, period):
    """❗重写：兼容 time/valid_time，输出 4 个 DataArray。"""
    import rioxarray  # noqa: F401

    ds = xr.open_dataset(sm_nc)
    da_list = []
    shp = shapefile_gdf.to_crs("EPSG:4326")
    for var in sm_vars:
        da = ds[var]
        if "valid_time" in da.coords:
            da = da.rename({"valid_time": "time"})
        da["time"] = da.indexes["time"].to_period("M").to_timestamp()
        da = da.sel(time=slice(period[0], period[-1]))
        da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(
            x_dim="longitude" if "longitude" in da.dims else "lon",
            y_dim="latitude" if "latitude" in da.dims else "lat",
        )
        da = da.rio.clip(shp.geometry, shp.crs, drop=True, all_touched=True)
        da_list.append(da)
    return da_list


def _da_to_stack_and_transform(da):
    """将 DataArray(time, y, x) 转为 numpy 3D，并构建对应 transform/crs。"""
    y_name = "latitude" if "latitude" in da.dims else "lat"
    x_name = "longitude" if "longitude" in da.dims else "lon"
    y = da[y_name].values
    x = da[x_name].values
    west, east = float(np.nanmin(x)), float(np.nanmax(x))
    south, north = float(np.nanmin(y)), float(np.nanmax(y))
    width, height = len(x), len(y)
    transform = from_bounds(west, south, east, north, width, height)
    stack = da.transpose("time", y_name, x_name).values
    crs = da.rio.crs if hasattr(da, "rio") else "EPSG:4326"
    return stack, transform, crs


def reproject_stack_to_match(src_stack, src_transform, src_crs, dst_shape, dst_transform, dst_crs, resampling=Resampling.bilinear):
    """将 3D 时序栅格重投影/重采样到目标网格。"""
    t, _, _ = src_stack.shape
    out = np.full((t, dst_shape[0], dst_shape[1]), np.nan, dtype=float)
    for i in range(t):
        dst = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(
            source=src_stack[i].astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        out[i] = dst
    return out


def load_spei03_series(spei_file: str, min_lon, max_lon, min_lat, max_lat, period):
    """✅可直接使用：读取单个 SPEI03 netCDF 的区域平均序列。"""
    ds = xr.open_dataset(spei_file)
    lat_values = ds["lat"].values
    lat_slice = slice(min_lat, max_lat) if lat_values[0] < lat_values[-1] else slice(max_lat, min_lat)
    spei03_series = ds["spei"].sel(lat=lat_slice, lon=slice(min_lon, max_lon)).mean(dim=["lat", "lon"]).to_pandas()
    spei03_series.index = pd.to_datetime(spei03_series.index)
    spei03_series.name = "spei03"
    if period is not None:
        spei03_series = spei03_series.loc[pd.to_datetime(period[0]): pd.to_datetime(period[-1])]
    return spei03_series


def identify_drought_events(spei_series: pd.Series, sm_series: pd.Series, spei_thr=-0.5, sm_quantile=0.2):
    """新增：基于 SPEI+土壤湿度识别干旱事件并输出事件表。"""
    df = pd.concat([spei_series.rename("spei"), sm_series.rename("sm")], axis=1).dropna()
    sm_thr = df["sm"].quantile(sm_quantile)
    drought = (df["spei"] <= spei_thr) & (df["sm"] <= sm_thr)

    events, in_evt, start = [], False, None
    prev = None
    for t, flag in drought.items():
        if flag and not in_evt:
            in_evt, start = True, t
        if (not flag) and in_evt:
            end = prev
            seg = df.loc[start:end]
            events.append({
                "start": start,
                "end": end,
                "duration_months": len(seg),
                "intensity_spei": float(seg["spei"].mean()),
                "intensity_sm": float(seg["sm"].mean()),
            })
            in_evt = False
        prev = t
    if in_evt and prev is not None:
        seg = df.loc[start:prev]
        events.append({"start": start, "end": prev, "duration_months": len(seg), "intensity_spei": float(seg["spei"].mean()), "intensity_sm": float(seg["sm"].mean())})
    return pd.DataFrame(events)


def compute_resistance_resilience(gpp_series: pd.Series, drought_events: pd.DataFrame):
    """新增：Resistance / Resilience / Recovery time 指标。"""
    out = []
    for _, evt in drought_events.iterrows():
        s, e = evt["start"], evt["end"]
        pre = gpp_series.loc[: s - pd.offsets.MonthBegin(1)].tail(12).mean()
        drought_mean = gpp_series.loc[s:e].mean()
        post = gpp_series.loc[e + pd.offsets.MonthBegin(1):].head(12).mean()

        resistance = np.nan if pd.isna(pre) or pre == 0 else drought_mean / pre
        resilience = np.nan if pd.isna(pre) or pre == 0 else post / pre

        threshold = 0.9 * pre
        rec_time = np.nan
        follow = gpp_series.loc[e + pd.offsets.MonthBegin(1):]
        hit = np.where(follow.values >= threshold)[0]
        if len(hit) > 0:
            rec_time = int(hit[0] + 1)

        out.append({"start": s, "end": e, "Resistance": resistance, "Resilience": resilience, "RecoveryTime": rec_time})
    return pd.DataFrame(out)


def map_igbp_to_major(igbp_array: np.ndarray) -> np.ndarray:
    """新增：将 IGBP 重分类为森林/灌丛/草地/农田四类。"""
    out = np.full(igbp_array.shape, "other", dtype=object)
    out[np.isin(igbp_array, [1, 2, 3, 4, 5])] = "forest"
    out[np.isin(igbp_array, [6, 7])] = "shrub"
    out[np.isin(igbp_array, [10])] = "grass"
    out[np.isin(igbp_array, [12, 14])] = "cropland"
    return out


def load_landcover_resampled(landcover_tif: Path, gpp_shape, gpp_transform, gpp_crs):
    """新增：把土地覆盖重投影到 GPP 网格。"""
    with rasterio.open(landcover_tif) as src:
        lc = src.read(1).astype(float)
        if src.nodata is not None:
            lc[lc == src.nodata] = np.nan
        dst = np.full(gpp_shape, np.nan, dtype=float)
        reproject(
            source=lc,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=gpp_transform,
            dst_crs=gpp_crs,
            resampling=Resampling.nearest,
        )
    return dst


def build_vegtype_timeseries(gpp_anom: np.ndarray, sm_anom: np.ndarray, times: pd.DatetimeIndex, lc_major: np.ndarray):
    """新增：生成按植被类型聚合的时序样本，用于类型差异 + XGBoost。"""
    records = []
    for veg in ["forest", "shrub", "grass", "cropland"]:
        m = lc_major == veg
        if np.nansum(m) == 0:
            continue
        gpp_v = np.nanmean(np.where(m[None, :, :], gpp_anom, np.nan), axis=(1, 2))
        sm_v = np.nanmean(np.where(m[None, :, :], sm_anom, np.nan), axis=(1, 2))
        records.append(pd.DataFrame({"time": times, "veg_type": veg, "gpp_anom": gpp_v, "sm_anom": sm_v}))
    if not records:
        return pd.DataFrame(columns=["time", "veg_type", "gpp_anom", "sm_anom"])
    return pd.concat(records, ignore_index=True)


def run_xgboost_attribution(df_features: pd.DataFrame, target_col: str = "gpp_anom"):
    """新增：XGBoost + 贝叶斯优化 + SHAP，返回逐变量重要性。"""
    xgb_mod = importlib.import_module("xgboost")
    shap_mod = importlib.import_module("shap")

    y = df_features[target_col].values
    X = df_features.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    model = xgb_mod.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        tree_method="hist",
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    # 优先使用贝叶斯优化
    if importlib.util.find_spec("skopt") is not None:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
        search = BayesSearchCV(
            pipe,
            {
                "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(2, 10),
                "model__reg_lambda": Real(1e-3, 10, prior="log-uniform"),
                "model__subsample": Real(0.5, 1.0),
                "model__colsample_bytree": Real(0.5, 1.0),
                "model__min_child_weight": Integer(1, 10),
            },
            n_iter=30,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
        optimizer_name = "BayesSearchCV"
    else:
        search = RandomizedSearchCV(
            pipe,
            {
                "model__learning_rate": np.linspace(0.01, 0.3, 15),
                "model__max_depth": np.arange(2, 11),
                "model__reg_lambda": np.logspace(-3, 1, 20),
                "model__subsample": np.linspace(0.5, 1.0, 10),
                "model__colsample_bytree": np.linspace(0.5, 1.0, 10),
                "model__min_child_weight": np.arange(1, 11),
            },
            n_iter=30,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1,
        )
        optimizer_name = "RandomizedSearchCV(fallback)"

    search.fit(X, y)
    best = search.best_estimator_
    pred = best.predict(X)

    # 特征名（数值列 + OneHot 后类别列）
    pre_fitted = best.named_steps["pre"]
    feature_names = pre_fitted.get_feature_names_out()

    # XGBoost 内置重要性
    gain_importance = best.named_steps["model"].feature_importances_

    # SHAP重要性
    X_trans = pre_fitted.transform(X)
    explainer = shap_mod.TreeExplainer(best.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)
    shap_mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": gain_importance,
        "importance_shap": shap_mean_abs,
    }).sort_values("importance_shap", ascending=False).reset_index(drop=True)

    return {
        "optimizer": optimizer_name,
        "best_params": search.best_params_,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "importance_df": importance_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="2000-2022 干旱-植被生产力分析流程",
        epilog=(
            "示例: python drought_gpp_pipeline.py --shapefile E:/.../云南省.shp "
            "--gpp_dir G:/FluxSat/FluxSat_GPP_2000_2022 --era5_path G:/2000-2022 ERA5.nc "
            "--spei_file G:/spei/spei03.nc --landcover_tif E:/.../MCD12Q1_IGBP_0p05deg_2010.tif"
        ),
    )
    # 推荐写法：显式参数名
    parser.add_argument("--shapefile", help="云南省边界shp")
    parser.add_argument("--gpp_dir", help="FluxSat GPP目录")
    parser.add_argument("--era5_path", help="ERA5-Land nc文件")
    parser.add_argument("--spei_file", help="SPEI03 nc文件")
    parser.add_argument("--landcover_tif", help="MCD12Q1 IGBP土地覆盖")

    # 兼容写法：位置参数（不需要 required）
    parser.add_argument("shapefile_pos", nargs="?")
    parser.add_argument("gpp_dir_pos", nargs="?")
    parser.add_argument("era5_path_pos", nargs="?")
    parser.add_argument("spei_file_pos", nargs="?")

    parser.add_argument("--start", default="2000-03-01")
    parser.add_argument("--end", default="2022-03-01")
    parser.add_argument("--gpp_pattern", default="{year}_{month:02d}_FluxSat.tif")
    parser.add_argument("--outdir", default="outputs")
    args = parser.parse_args()

    args.shapefile = args.shapefile or args.shapefile_pos
    args.gpp_dir = args.gpp_dir or args.gpp_dir_pos
    args.era5_path = args.era5_path or args.era5_path_pos
    args.spei_file = args.spei_file or args.spei_file_pos

    missing = [k for k in ["shapefile", "gpp_dir", "era5_path", "spei_file"] if getattr(args, k) is None]
    if missing:
        parser.error(f"缺少必要参数: {', '.join(missing)}")
    return args


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    shapefile = gpd.read_file(args.shapefile)
    period = month_range(args.start, args.end)
    min_lon, min_lat, max_lon, max_lat = shapefile.total_bounds

    gpp_stack, gpp_transform, gpp_crs, _ = load_gpp_stack(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)
    sm1, sm2, sm3, sm4 = load_sm_era5_stack(args.era5_path, ["swvl1", "swvl2", "swvl3", "swvl4"], shapefile, period)
    sm_era5_da = compute_sm_era5_xr(sm1, sm2, sm3, sm4)
    sm_era5_stack, sm_transform, sm_crs = _da_to_stack_and_transform(sm_era5_da)

    # 关键修复：将 ERA5 土壤湿度统一到 GPP 网格，避免与 landcover/GPP 维度不一致
    sm_era5 = reproject_stack_to_match(
        sm_era5_stack,
        sm_transform,
        sm_crs,
        gpp_stack.shape[1:],
        gpp_transform,
        gpp_crs,
        resampling=Resampling.bilinear,
    )

    gpp_anom = compute_pixelwise_detrended_anomaly(gpp_stack, period)
    sm_anom = compute_pixelwise_detrended_anomaly(sm_era5, period)

    gpp_mean_series = pd.Series(np.nanmean(gpp_anom.reshape(len(period), -1), axis=1), index=period, name="gpp_anom")
    sm_mean_series = pd.Series(np.nanmean(sm_anom.reshape(len(period), -1), axis=1), index=period, name="sm_anom")
    spei03_series = load_spei03_series(args.spei_file, min_lon, max_lon, min_lat, max_lat, period)

    drought_events = identify_drought_events(spei03_series, sm_mean_series)
    drought_events.to_csv(outdir / "drought_events.csv", index=False, encoding="utf-8-sig")

    rr = compute_resistance_resilience(gpp_mean_series, drought_events)
    rr.to_csv(outdir / "resistance_resilience.csv", index=False, encoding="utf-8-sig")

    if args.landcover_tif:
        lc = load_landcover_resampled(Path(args.landcover_tif), gpp_stack.shape[1:], gpp_transform, gpp_crs)
        lc_major = map_igbp_to_major(lc)
        veg_ts = build_vegtype_timeseries(gpp_anom, sm_anom, period, lc_major)
        veg_ts = veg_ts.merge(spei03_series.rename("spei").reset_index(names="time"), on="time", how="left")
        veg_ts["sm_curr"] = veg_ts["sm_anom"]
        veg_ts["spei_curr"] = veg_ts["spei"]
        veg_ts["sm_lag1"] = veg_ts.groupby("veg_type")["sm_anom"].shift(1)
        veg_ts["spei_lag1"] = veg_ts.groupby("veg_type")["spei"].shift(1)
        veg_ts = veg_ts.dropna(subset=["gpp_anom", "sm_curr", "spei_curr", "sm_lag1", "spei_lag1"])
        veg_ts.to_csv(outdir / "veg_type_timeseries.csv", index=False, encoding="utf-8-sig")
        df_model = veg_ts[["gpp_anom", "sm_curr", "spei_curr", "sm_lag1", "spei_lag1", "veg_type"]].copy()
    else:
        df_model = pd.concat([
            gpp_mean_series.rename("gpp_anom"),
            sm_mean_series.rename("sm_curr"),
            spei03_series.rename("spei_curr"),
            sm_mean_series.shift(1).rename("sm_lag1"),
            spei03_series.shift(1).rename("spei_lag1"),
        ], axis=1).dropna()
        df_model["veg_type"] = "mixed"

    result = run_xgboost_attribution(df_model, target_col="gpp_anom")
    with open(outdir / "xgboost_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"optimizer={result['optimizer']}\n")
        f.write(f"best_params={result['best_params']}\n")
        f.write(f"rmse={result['rmse']:.4f}\n")
        f.write(f"r2={result['r2']:.4f}\n")

    # 输出每个变量的重要程度
    result["importance_df"].to_csv(outdir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))
    plt.plot(gpp_mean_series.index, gpp_mean_series.values, label="GPP anomaly")
    plt.plot(sm_mean_series.index, sm_mean_series.values, label="SM anomaly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "timeseries_overview.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main(parse_args())
