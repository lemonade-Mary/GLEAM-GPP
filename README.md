# GLEAM-GPP

云南省干旱-植被多源联合分析代码。

## 主脚本

- `yunnan_drought_analysis.py`：整合版脚本，已把以下需求统一到一个 `.py` 中：
  1. 土壤水分与 GPP 的当月去趋势异常时间序列；
  2. 2009-2010 干旱最严重区域识别及该区域 GPP 异常对比；
  3. 按 IGBP 土地利用类型分析植被恢复时间；
  4. 引入 NDVI/EVI、GOSIF 做同步性对比；
  5. 引入 SPEI（12个月 nc）并进行 2009-2015 连续小干旱机器学习影响分析。

## 依赖

```bash
pip install geopandas rasterio xarray scipy numpy pandas matplotlib seaborn scikit-learn affine
```

并确保可用：`from osgeo import gdal`（用于 MOD13C2 HDF 读取）。

## 用法示例

```bash
python yunnan_drought_analysis.py \
  --shapefile "E:/大创/省份边界shp/云南省.shp" \
  --gpp-dir "G:/数据/FluxSat/FluxSat_GPP_2000_2022" \
  --sm-nc "D:/GLEAM/v3.8a/SMroot_1980-2022_GLEAM_v3.8a_MO.nc" \
  --landcover "E:/大创/MCD12Q1_IGBP_0p05deg_2010.tif" \
  --modis-dir "G:/modis" \
  --gosif-dir "G:/数据/Orig" \
  --spei-dir "D:/spei" \
  --outdir "outputs"
```
