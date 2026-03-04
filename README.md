# GLEAM-GPP

用于云南省土壤水分（GLEAM）与 GPP（FluxSat）联合干旱分析。

## 文件

- `yunnan_drought_analysis.py`：脚本版流程（CLI 运行）。
- `yunnan_drought_analysis_notebook.ipynb`：Notebook 分块版流程，包含 NDVI/EVI、GOSIF、SPEI 与 2009-2015 连续小干旱机器学习分析。

## Notebook 主要内容

1. 云南省土壤水分与 GPP 当月去趋势异常时间序列（带坐标和单位）。
2. NDVI/EVI（MOD13C2）与 GOSIF 异常计算，并与干旱指标做同步性对比。
3. SPEI 月尺度子图（12个月）及区域时间序列融合。
4. 机器学习模型比较（Linear/Ridge/RF/GBDT/SVR），分析连续小干旱对 GPP 异常影响。

## 依赖

```bash
pip install geopandas rasterio xarray scipy numpy pandas matplotlib seaborn scikit-learn
```

另需可用的 GDAL Python 绑定（`from osgeo import gdal`）。
