# GLEAM-GPP

用于云南省土壤水分（GLEAM）与 GPP（FluxSat）联合干旱分析的脚本仓库。

## 脚本

- `yunnan_drought_analysis.py`：整理后的完整流程，输出 3 类图件与统计表：
  1. 云南省土壤水分与 GPP 的**当月去趋势异常**时间序列图；
  2. 2009-2010 年严重干旱区域识别（土壤水分）及该区域 GPP 异常对比图；
  3. 按 IGBP 土地利用类型统计植被恢复时间箱线图与汇总表。

## 依赖

```bash
pip install geopandas rasterio xarray scipy numpy pandas matplotlib seaborn affine
```

## 用法

```bash
python yunnan_drought_analysis.py \
  --shapefile "E:/大创/省份边界shp/云南省.shp" \
  --gpp-dir "G:/数据/FluxSat/FluxSat_GPP_2000_2022" \
  --sm-nc "D:/GLEAM/v3.8a/SMroot_1980-2022_GLEAM_v3.8a_MO.nc" \
  --landcover "E:/大创/MCD12Q1_IGBP_0p05deg_2010.tif" \
  --outdir "outputs"
```

输出文件默认保存在 `outputs/`。
