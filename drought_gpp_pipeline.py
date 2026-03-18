import argparse
from pathlib import Path
import importlib
from osgeo import gdal

import geopandas as gpd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rasterio
import networkx as nx
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
from sklearn.model_selection import cross_validate

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

def compute_vpd_from_t2m_dewpoint(t2m_da, d2m_da):
    """根据ERA5 2m气温和露点温度计算VPD（kPa）。"""
    t_c = t2m_da - 273.15
    td_c = d2m_da - 273.15
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    vpd = es - ea
    vpd.name = "vpd"
    return vpd


def load_era5_variable(era5_nc, var_name, shapefile_gdf, period):
    """读取ERA5单变量，裁剪到研究区并统一时间维。"""
    import rioxarray  # noqa: F401

    ds = xr.open_dataset(era5_nc)
    da = ds[var_name]
    if "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})
    da["time"] = da.indexes["time"].to_period("M").to_timestamp()
    da = da.sel(time=slice(period[0], period[-1]))
    shp = shapefile_gdf.to_crs("EPSG:4326")
    da = da.rio.write_crs("EPSG:4326").rio.set_spatial_dims(
        x_dim="longitude" if "longitude" in da.dims else "lon",
        y_dim="latitude" if "latitude" in da.dims else "lat",
    )
    da = da.rio.clip(shp.geometry, shp.crs, drop=True, all_touched=True)
    return da


def load_gosif_stack(gosif_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex, gosif_pattern: str):
    """读取并裁剪GOSIF月尺度栅格。"""
    stack = []
    transform, crs, nodata = None, None, None
    for t in period:
        path = gosif_dir / gosif_pattern.format(year=t.year, month=t.month)
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
        raise ValueError("未读取到任何 GOSIF 栅格。")
    for i, arr in enumerate(stack):
        if arr is None:
            stack[i] = np.full_like(template, np.nan, dtype=float)
    return np.stack(stack), transform, crs, nodata


def _read_modis_cmg_layer(hdf_path: Path, subdataset_name: str):
    ds = gdal.Open(f'HDF4_EOS:EOS_GRID:"{hdf_path}":MOD_Grid_monthly_CMG_VI:{subdataset_name}')
    if ds is None:
        raise ValueError(f"无法读取 MODIS 子数据集: {hdf_path.name} -> {subdataset_name}")
    arr = ds.ReadAsArray().astype(float)
    arr[arr == -3000] = np.nan
    arr *= 0.0001
    gt = ds.GetGeoTransform()
    transform = from_origin(gt[0], gt[3], gt[1], abs(gt[5]))
    return arr, transform


def load_modis_evi_stack(modis_dir: Path, shapefile: gpd.GeoDataFrame, period: pd.DatetimeIndex):
    """读取MODIS MOD13C2 EVI，并裁剪到研究区。"""
    stack = []
    transform, crs = None, "EPSG:4326"
    shp = shapefile.to_crs("EPSG:4326")
    shapes = [shp.geometry.unary_union]
    for t in period:
        path = modis_dir / f"MOD13C2.A{t.year:04d}.M{t.month:02d}.hdf"
        if not path.exists():
            stack.append(None)
            continue
        arr, src_transform = _read_modis_cmg_layer(path, "CMG 0.05 Deg Monthly EVI")
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=src_transform,
                nodata=np.nan,
            ) as dataset:
                dataset.write(arr.astype(np.float32), 1)
                out_image, out_transform = mask(dataset, shapes, crop=True)
                clipped = out_image[0].astype(float)
                clipped[~np.isfinite(clipped)] = np.nan
                if transform is None:
                    transform = out_transform
                stack.append(clipped)
    template = next((x for x in stack if x is not None), None)
    if template is None:
        raise ValueError("未读取到任何 MODIS EVI 数据。")
    for i, arr in enumerate(stack):
        if arr is None:
            stack[i] = np.full_like(template, np.nan, dtype=float)
    return np.stack(stack), transform, crs



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



def save_array_map(arr, title: str, out_png: Path, cmap="RdYlBu_r", cbar_label=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("列")
    ax.set_ylabel("行")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def aggregate_by_vegtype(stack: np.ndarray, lc_major: np.ndarray, times: pd.DatetimeIndex, col_name: str):
    records = []
    for veg in ["forest", "shrub", "grass", "cropland"]:
        m = lc_major == veg
        if np.nansum(m) == 0:
            continue
        series = np.nanmean(np.where(m[None, :, :], stack, np.nan), axis=(1, 2))
        records.append(pd.DataFrame({"time": times, "veg_type": veg, col_name: series}))
    if not records:
        return pd.DataFrame(columns=["time", "veg_type", col_name])
    return pd.concat(records, ignore_index=True)


def classify_drought_pixels(spei_series: pd.Series, threshold=-0.5):
    return spei_series <= threshold


def compute_drought_heatmap(sm_anom: np.ndarray, spei_series: pd.Series, threshold=-0.5):
    drought_mask = classify_drought_pixels(spei_series, threshold).reindex(spei_series.index).values
    ntime, nrow, ncol = sm_anom.shape
    spatial_mean = sm_anom.reshape(ntime, -1)
    spatial_order = np.argsort(np.nanmean(spatial_mean, axis=0))
    heat = spatial_mean[:, spatial_order].T
    heat[:, ~drought_mask] = np.nan
    return heat


def build_continuous_drought_graph(drought_events: pd.DataFrame, max_gap_months=3):
    G = nx.Graph()
    if drought_events.empty:
        return G
    events = drought_events.copy()
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])
    for i, row in events.iterrows():
        G.add_node(i, start=row["start"], end=row["end"], duration=row["duration_months"])
    for i in range(len(events) - 1):
        gap = (events.iloc[i + 1]["start"].to_period("M") - events.iloc[i]["end"].to_period("M")).n
        if gap < max_gap_months:
            G.add_edge(events.index[i], events.index[i + 1], gap=gap)
    return G


def summarize_event_pixel_response(gpp_anom: np.ndarray, drought_mask_series: pd.Series):
    drought_mask = drought_mask_series.reindex(drought_mask_series.index).values
    drought_mean = np.nanmean(gpp_anom[drought_mask], axis=0)
    normal_mean = np.nanmean(gpp_anom[~drought_mask], axis=0)
    diff = drought_mean - normal_mean
    yearly_neg_ratio = {}
    years = np.unique(drought_mask_series.index.year)
    for year in years:
        idx = (drought_mask_series.index.year == year) & drought_mask
        if idx.sum() == 0:
            continue
        year_mean = np.nanmean(gpp_anom[idx], axis=0)
        yearly_neg_ratio[year] = float(np.nanmean(year_mean < 0))
    heterogeneity = np.nanstd(gpp_anom[drought_mask], axis=0)
    return diff, yearly_neg_ratio, heterogeneity


def compute_veg_event_box(veg_ts: pd.DataFrame, drought_events: pd.DataFrame):
    if veg_ts.empty or drought_events.empty:
        return pd.DataFrame(columns=["veg_type", "event_id", "gpp_anom"])
    out = []
    for evt_id, evt in drought_events.reset_index(drop=True).iterrows():
        mask = (veg_ts["time"] >= pd.to_datetime(evt["start"])) & (veg_ts["time"] <= pd.to_datetime(evt["end"]))
        tmp = veg_ts.loc[mask, ["veg_type", "gpp_anom"]].copy()
        tmp["event_id"] = evt_id
        tmp["intensity_bin"] = pd.cut([evt["intensity_spei"]] * len(tmp), bins=[-np.inf, -1.5, -1.0, -0.5, np.inf], labels=["extreme", "severe", "moderate", "mild"])
        out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["veg_type", "event_id", "gpp_anom", "intensity_bin"])


def build_recovery_trajectories(veg_ts: pd.DataFrame, drought_events: pd.DataFrame, max_months=12):
    rows = []
    if veg_ts.empty or drought_events.empty:
        return pd.DataFrame(columns=["veg_type", "sequence_id", "recovery_month", "relative_gpp"])
    drought_events = drought_events.copy().reset_index(drop=True)
    drought_events["sequence_id"] = 1
    for i in range(1, len(drought_events)):
        gap = (pd.to_datetime(drought_events.loc[i, "start"]).to_period("M") - pd.to_datetime(drought_events.loc[i - 1, "end"]).to_period("M")).n
        drought_events.loc[i, "sequence_id"] = drought_events.loc[i - 1, "sequence_id"] + (gap >= 3)
    for _, evt in drought_events.iterrows():
        end = pd.to_datetime(evt["end"])
        for veg, group in veg_ts.groupby("veg_type"):
            pre = group.loc[group["time"] < pd.to_datetime(evt["start"]), "gpp_anom"].tail(12).mean()
            if pd.isna(pre) or pre == 0:
                continue
            post = group.loc[group["time"] > end].head(max_months).copy()
            if post.empty:
                continue
            post["recovery_month"] = np.arange(1, len(post) + 1)
            post["relative_gpp"] = post["gpp_anom"] / pre
            post["veg_type"] = veg
            post["sequence_id"] = int(evt["sequence_id"])
            rows.append(post[["veg_type", "sequence_id", "recovery_month", "relative_gpp"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["veg_type", "sequence_id", "recovery_month", "relative_gpp"])


def plot_figure1(outdir: Path, dem_arr, spei_series: pd.Series, drought_heat, lc_major: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    im0 = axes[0, 0].imshow(dem_arr, cmap="terrain")
    axes[0, 0].set_title("(a) 研究区地形阴影")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    axes[0, 1].plot(spei_series.index, spei_series.values, color="brown")
    axes[0, 1].axhline(-0.5, ls="--", color="red")
    axes[0, 1].set_title("(b) 2000-2022年区域平均SPEI")
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[0, 1].tick_params(axis="x", rotation=30)

    im2 = axes[1, 0].imshow(drought_heat, aspect="auto", cmap="RdBu_r")
    axes[1, 0].set_title("(c) 干旱事件时空分布热力图")
    axes[1, 0].set_xlabel("时间")
    axes[1, 0].set_ylabel("空间位置(排序后像元)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    veg_code = {"other": 0, "forest": 1, "shrub": 2, "grass": 3, "cropland": 4}
    veg_numeric = np.vectorize(lambda x: veg_code.get(x, 0))(lc_major)
    im3 = axes[1, 1].imshow(veg_numeric, cmap="Set2", vmin=0, vmax=4)
    axes[1, 1].set_title("(d) 主要植被类型空间分布")
    cbar = fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["other", "forest", "shrub", "grass", "cropland"])

    fig.tight_layout()
    fig.savefig(outdir / "Figure1_study_area_overview.png", dpi=300)
    plt.close(fig)


def plot_figure2(outdir: Path, drought_events: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    if drought_events.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No drought events", ha="center", va="center")
            ax.axis("off")
    else:
        events = drought_events.copy().reset_index(drop=True)
        events["start"] = pd.to_datetime(events["start"])
        events["end"] = pd.to_datetime(events["end"])
        for i, row in events.iterrows():
            axes[0, 0].barh(i, (row["end"] - row["start"]).days + 30, left=row["start"], color="tomato")
        axes[0, 0].set_title("(a) 干旱事件时间轴")
        axes[0, 0].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[0, 0].tick_params(axis="x", rotation=30)

        axes[0, 1].boxplot([events["intensity_spei"].dropna(), events["intensity_sm"].dropna()], labels=["SPEI", "SM"])
        axes[0, 1].set_title("(b) 干旱事件强度分布")

        axes[1, 0].scatter(events["duration_months"], events["intensity_spei"], c=events["intensity_sm"], cmap="viridis")
        axes[1, 0].set_title("(c) 持续时间与强度")
        axes[1, 0].set_xlabel("持续时间(月)")
        axes[1, 0].set_ylabel("SPEI强度")

        G = build_continuous_drought_graph(events)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=axes[1, 1], with_labels=True, node_color="orange", edge_color="gray")
        axes[1, 1].set_title("(d) 连续干旱事件序列")
    fig.tight_layout()
    fig.savefig(outdir / "Figure2_drought_event_statistics.png", dpi=300)
    plt.close(fig)


def plot_figure3(outdir: Path, gpp_diff_map: np.ndarray, yearly_neg_ratio: dict, gpp_mean_series: pd.Series, veg_gpp_ts: pd.DataFrame, heterogeneity: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    im0 = axes[0, 0].imshow(gpp_diff_map, cmap="RdBu")
    axes[0, 0].set_title("(a) 干旱期 vs 非干旱期 GPP差异")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    if yearly_neg_ratio:
        years = list(yearly_neg_ratio.keys())
        vals = list(yearly_neg_ratio.values())
        axes[0, 1].plot(years, vals, marker="o")
    axes[0, 1].set_title("(b) 年际GPP负异常像元比例")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("负异常比例")

    axes[1, 0].plot(gpp_mean_series.index, gpp_mean_series.values, label="regional")
    if not veg_gpp_ts.empty:
        for veg, group in veg_gpp_ts.groupby("veg_type"):
            axes[1, 0].plot(group["time"], group["gpp_anom"], label=veg)
    axes[1, 0].set_title("(c) 典型干旱事件GPP时间序列")
    axes[1, 0].legend()

    im3 = axes[1, 1].imshow(heterogeneity, cmap="magma")
    axes[1, 1].set_title("(d) GPP响应空间异质性")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "Figure3_drought_gpp_spatiotemporal_patterns.png", dpi=300)
    plt.close(fig)


def plot_figure4(outdir: Path, event_box_df: pd.DataFrame, rr_veg: pd.DataFrame, interaction_df: pd.DataFrame, recovery_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    if not event_box_df.empty:
        data = [event_box_df.loc[event_box_df["veg_type"] == veg, "gpp_anom"].dropna() for veg in ["forest", "shrub", "grass", "cropland"]]
        axes[0, 0].boxplot(data, labels=["forest", "shrub", "grass", "cropland"])
    axes[0, 0].set_title("(a) 各植被类型干旱期GPP变化")

    if not rr_veg.empty:
        for veg, group in rr_veg.groupby("veg_type"):
            axes[0, 1].scatter(group["Resistance"], group["Resilience"], label=veg)
        axes[0, 1].legend()
    axes[0, 1].set_title("(b) Resistance vs Resilience")
    axes[0, 1].set_xlabel("Resistance")
    axes[0, 1].set_ylabel("Resilience")

    if not interaction_df.empty:
        pivot = interaction_df.pivot(index="intensity_bin", columns="veg_type", values="gpp_decline_ratio")
        im = axes[1, 0].imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        axes[1, 0].set_xticks(range(len(pivot.columns)), pivot.columns)
        axes[1, 0].set_yticks(range(len(pivot.index)), pivot.index)
        axes[1, 0].set_title("(c) 植被类型×干旱强度交互")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    if not recovery_df.empty:
        for (seq, veg), group in recovery_df.groupby(["sequence_id", "veg_type"]):
            axes[1, 1].plot(group["recovery_month"], group["relative_gpp"], label=f"{veg}-seq{seq}")
        axes[1, 1].legend(fontsize=8, ncol=2)
    axes[1, 1].set_title("(d) 不同干旱历史下恢复轨迹")
    axes[1, 1].set_xlabel("恢复月份")
    axes[1, 1].set_ylabel("相对GPP")

    fig.tight_layout()
    fig.savefig(outdir / "Figure4_vegetation_type_differences.png", dpi=300)
    plt.close(fig)


def plot_figure5(outdir: Path, df_model: pd.DataFrame, result: dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    xgb_mod = importlib.import_module("xgboost")
    X = df_model.drop(columns=["gpp_anom"])
    y = df_model["gpp_anom"]
    X_enc = pd.get_dummies(X)
    model = xgb_mod.XGBRegressor(objective="reg:squarederror", n_estimators=200, tree_method="hist", random_state=42)
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_validate(model, X_enc, y, cv=cv, scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
    axes[0, 0].plot(range(1, len(scores["test_r2"]) + 1), scores["test_r2"], marker="o", label="R2")
    axes[0, 0].plot(range(1, len(scores["test_rmse"]) + 1), -scores["test_rmse"], marker="s", label="RMSE")
    axes[0, 0].set_title("(a) XGBoost模型性能")
    axes[0, 0].legend()

    imp = result["importance_df"].head(15).iloc[::-1]
    axes[0, 1].barh(imp["feature"], imp["importance_gain"], alpha=0.7, label="Gain")
    axes[0, 1].barh(imp["feature"], imp["importance_shap"], alpha=0.7, label="SHAP")
    axes[0, 1].set_title("(b) 特征重要性排序")
    axes[0, 1].legend()

    top_feats = result["importance_df"].head(5)["feature"].tolist()
    for feat in top_feats:
        sub = result["importance_df"].loc[result["importance_df"]["feature"] == feat]
        axes[1, 0].scatter([feat], sub["importance_shap"], s=60)
    axes[1, 0].set_title("(c) SHAP值散点概览")
    axes[1, 0].tick_params(axis="x", rotation=30)

    dep_feats = [f for f in result["importance_df"]["feature"] if any(key in f for key in ["sm_", "vpd", "spei"])][:3]
    for feat in dep_feats:
        if feat in X_enc.columns:
            axes[1, 1].scatter(X_enc[feat], np.repeat(result["importance_df"].set_index("feature").loc[feat, "importance_shap"], len(X_enc)), s=8, alpha=0.5, label=feat)
    axes[1, 1].set_title("(d) 关键特征SHAP依赖图(近似)")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "Figure5_xgboost_attribution.png", dpi=300)
    plt.close(fig)

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
    parser.add_argument("--modis_dir", help="MOD13C2 EVI目录")
    parser.add_argument("--gosif_dir", help="GOSIF目录")
    parser.add_argument("--gosif_pattern", default="GOSIF_{year:04d}.M{month:02d}.tif", help="GOSIF文件名模板")

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

    # ======================
    # 1️⃣ 数据读取并统一到GPP网格
    # ======================
    gpp_stack, gpp_transform, gpp_crs, _ = load_gpp_stack(Path(args.gpp_dir), shapefile, period, args.gpp_pattern)

    sm1, sm2, sm3, sm4 = load_sm_era5_stack(
        args.era5_path, ["swvl1", "swvl2", "swvl3", "swvl4"], shapefile, period
    )
    sm_era5_da = compute_sm_era5_xr(sm1, sm2, sm3, sm4)
    sm_era5_stack, sm_transform, sm_crs = _da_to_stack_and_transform(sm_era5_da)
    sm_era5 = reproject_stack_to_match(
        sm_era5_stack, sm_transform, sm_crs, gpp_stack.shape[1:], gpp_transform, gpp_crs, resampling=Resampling.bilinear
    )

    t2m_da = load_era5_variable(args.era5_path, "t2m", shapefile, period)
    d2m_da = load_era5_variable(args.era5_path, "d2m", shapefile, period)
    vpd_da = compute_vpd_from_t2m_dewpoint(t2m_da, d2m_da)
    vpd_stack, vpd_transform, vpd_crs = _da_to_stack_and_transform(vpd_da)
    vpd = reproject_stack_to_match(
        vpd_stack, vpd_transform, vpd_crs, gpp_stack.shape[1:], gpp_transform, gpp_crs, resampling=Resampling.bilinear
    )

    if not args.modis_dir or not args.gosif_dir:
        raise ValueError("请提供 --modis_dir 和 --gosif_dir，使 EVI/GOSIF 进入 XGBoost 模型。")

    evi_stack, evi_transform, evi_crs = load_modis_evi_stack(Path(args.modis_dir), shapefile, period)
    evi = reproject_stack_to_match(
        evi_stack, evi_transform, evi_crs, gpp_stack.shape[1:], gpp_transform, gpp_crs, resampling=Resampling.bilinear
    )

    gosif_stack, gosif_transform, gosif_crs, _ = load_gosif_stack(Path(args.gosif_dir), shapefile, period, args.gosif_pattern)
    gosif = reproject_stack_to_match(
        gosif_stack, gosif_transform, gosif_crs, gpp_stack.shape[1:], gpp_transform, gpp_crs, resampling=Resampling.bilinear
    )

    # ======================
    # 2️⃣ 计算异常
    # ======================
    gpp_anom = compute_pixelwise_detrended_anomaly(gpp_stack, period)
    sm_anom = compute_pixelwise_detrended_anomaly(sm_era5, period)
    vpd_anom = compute_pixelwise_detrended_anomaly(vpd, period)
    evi_anom = compute_pixelwise_detrended_anomaly(evi, period)
    gosif_anom = compute_pixelwise_detrended_anomaly(gosif, period)

    # ======================
    # 3️⃣ 区域平均时间序列
    # ======================
    gpp_mean_series = pd.Series(np.nanmean(gpp_anom.reshape(len(period), -1), axis=1), index=period, name="gpp_anom")
    sm_mean_series = pd.Series(np.nanmean(sm_anom.reshape(len(period), -1), axis=1), index=period, name="sm_anom")
    vpd_mean_series = pd.Series(np.nanmean(vpd_anom.reshape(len(period), -1), axis=1), index=period, name="vpd_anom")
    evi_mean_series = pd.Series(np.nanmean(evi_anom.reshape(len(period), -1), axis=1), index=period, name="evi_anom")
    gosif_mean_series = pd.Series(np.nanmean(gosif_anom.reshape(len(period), -1), axis=1), index=period, name="gosif_anom")
    spei03_series = load_spei03_series(args.spei_file, min_lon, max_lon, min_lat, max_lat, period)

    drought_events = identify_drought_events(spei03_series, sm_mean_series)
    drought_events.to_csv(outdir / "drought_events.csv", index=False, encoding="utf-8-sig")

    rr = compute_resistance_resilience(gpp_mean_series, drought_events)
    rr.to_csv(outdir / "resistance_resilience.csv", index=False, encoding="utf-8-sig")

    # ======================
    # 4️⃣ 构建建模数据
    # ======================
    if args.landcover_tif:
        lc = load_landcover_resampled(Path(args.landcover_tif), gpp_stack.shape[1:], gpp_transform, gpp_crs)
        lc_major = map_igbp_to_major(lc)
        veg_ts = build_vegtype_timeseries(gpp_anom, sm_anom, period, lc_major)
        for col_name, arr in [("vpd_anom", vpd_anom), ("evi_anom", evi_anom), ("gosif_anom", gosif_anom)]:
            vals = []
            for veg in ["forest", "shrub", "grass", "cropland"]:
                m = lc_major == veg
                if np.nansum(m) == 0:
                    continue
                vals.append(pd.DataFrame({"time": period, "veg_type": veg, col_name: np.nanmean(np.where(m[None, :, :], arr, np.nan), axis=(1, 2))}))
            if vals:
                veg_ts = veg_ts.merge(pd.concat(vals, ignore_index=True), on=["time", "veg_type"], how="left")
        veg_ts = veg_ts.merge(spei03_series.rename("spei").reset_index(names="time"), on="time", how="left")
        veg_ts["sm_curr"] = veg_ts["sm_anom"]
        veg_ts["spei_curr"] = veg_ts["spei"]
        veg_ts["vpd_curr"] = veg_ts["vpd_anom"]
        veg_ts["evi_curr"] = veg_ts["evi_anom"]
        veg_ts["gosif_curr"] = veg_ts["gosif_anom"]
        veg_ts["sm_lag1"] = veg_ts.groupby("veg_type")["sm_anom"].shift(1)
        veg_ts["spei_lag1"] = veg_ts.groupby("veg_type")["spei"].shift(1)
        veg_ts["vpd_lag1"] = veg_ts.groupby("veg_type")["vpd_anom"].shift(1)
        veg_ts["evi_lag1"] = veg_ts.groupby("veg_type")["evi_anom"].shift(1)
        veg_ts["gosif_lag1"] = veg_ts.groupby("veg_type")["gosif_anom"].shift(1)
        veg_ts = veg_ts.dropna(subset=[
            "gpp_anom", "sm_curr", "spei_curr", "vpd_curr", "evi_curr", "gosif_curr",
            "sm_lag1", "spei_lag1", "vpd_lag1", "evi_lag1", "gosif_lag1"
        ])
        veg_ts.to_csv(outdir / "veg_type_timeseries.csv", index=False, encoding="utf-8-sig")
        df_model = veg_ts[[
            "gpp_anom", "sm_curr", "spei_curr", "vpd_curr", "evi_curr", "gosif_curr",
            "sm_lag1", "spei_lag1", "vpd_lag1", "evi_lag1", "gosif_lag1", "veg_type"
        ]].copy()
    else:
        df_model = pd.concat([
            gpp_mean_series.rename("gpp_anom"),
            sm_mean_series.rename("sm_curr"),
            spei03_series.rename("spei_curr"),
            vpd_mean_series.rename("vpd_curr"),
            evi_mean_series.rename("evi_curr"),
            gosif_mean_series.rename("gosif_curr"),
            sm_mean_series.shift(1).rename("sm_lag1"),
            spei03_series.shift(1).rename("spei_lag1"),
            vpd_mean_series.shift(1).rename("vpd_lag1"),
            evi_mean_series.shift(1).rename("evi_lag1"),
            gosif_mean_series.shift(1).rename("gosif_lag1"),
        ], axis=1).dropna()
        df_model["veg_type"] = "mixed"

    result = run_xgboost_attribution(df_model, target_col="gpp_anom")
    with open(outdir / "xgboost_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"optimizer={result['optimizer']}\n")
        f.write(f"best_params={result['best_params']}\n")
        f.write(f"rmse={result['rmse']:.4f}\n")
        f.write(f"r2={result['r2']:.4f}\n")

    result["importance_df"].to_csv(outdir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))
    plt.plot(gpp_mean_series.index, gpp_mean_series.values, label="GPP anomaly")
    plt.plot(sm_mean_series.index, sm_mean_series.values, label="SM anomaly")
    plt.plot(vpd_mean_series.index, vpd_mean_series.values, label="VPD anomaly")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "timeseries_overview.png", dpi=300)
    plt.close()

    drought_mask_series = classify_drought_pixels(spei03_series)
    gpp_diff_map, yearly_neg_ratio, heterogeneity = summarize_event_pixel_response(gpp_anom, drought_mask_series)

    if args.landcover_tif:
        dem_arr = np.where(np.isfinite(lc), lc, np.nan)
        veg_gpp_ts = veg_ts[["time", "veg_type", "gpp_anom"]].copy()
        event_box_df = compute_veg_event_box(veg_ts, drought_events)
        rr_veg = []
        for veg, group in veg_ts.groupby("veg_type"):
            tmp_rr = compute_resistance_resilience(group.set_index("time")["gpp_anom"], drought_events)
            if not tmp_rr.empty:
                tmp_rr["veg_type"] = veg
                rr_veg.append(tmp_rr)
        rr_veg = pd.concat(rr_veg, ignore_index=True) if rr_veg else pd.DataFrame(columns=["veg_type", "Resistance", "Resilience"])
        if not event_box_df.empty:
            interaction_df = event_box_df.groupby(["veg_type", "intensity_bin"], observed=False)["gpp_anom"].mean().reset_index()
            interaction_df["gpp_decline_ratio"] = -interaction_df["gpp_anom"]
        else:
            interaction_df = pd.DataFrame(columns=["veg_type", "intensity_bin", "gpp_decline_ratio"])
        recovery_df = build_recovery_trajectories(veg_ts, drought_events)
        drought_heat = compute_drought_heatmap(sm_anom, spei03_series)
        plot_figure1(outdir, dem_arr, spei03_series, drought_heat, lc_major)
        plot_figure4(outdir, event_box_df, rr_veg, interaction_df, recovery_df)
    else:
        dem_arr = np.nanmean(gpp_stack, axis=0)
        veg_gpp_ts = pd.DataFrame(columns=["time", "veg_type", "gpp_anom"])
        drought_heat = compute_drought_heatmap(sm_anom, spei03_series)
        plot_figure1(outdir, dem_arr, spei03_series, drought_heat, np.full(gpp_stack.shape[1:], "other", dtype=object))

    plot_figure2(outdir, drought_events)
    plot_figure3(outdir, gpp_diff_map, yearly_neg_ratio, gpp_mean_series, veg_gpp_ts, heterogeneity)
    plot_figure5(outdir, df_model, result)


if __name__ == "__main__":
    main(parse_args())
