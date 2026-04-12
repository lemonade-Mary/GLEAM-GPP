# Drought Resilience 3D Framework (Resistance–Recovery–Adaptability)

本仓库提供一个可直接复现论文方法的 Python 全流程代码，实现：

1. 数据预处理（重采样到统一分辨率、标准化、GPP 去季节/去趋势）；
2. 有效干旱事件识别（MCDI<=-1、至少3个月、且存在 GPP 负异常）；
3. Resistance / Recovery / Adaptability 计算；
4. 熵权法综合 Resilience；
5. AR(1) 验证；
6. XGBoost + SHAP 驱动因子分析。

---

## 文件结构

- `preprocess.py`：预处理模块。
- `drought_detection.py`：干旱事件识别。
- `resilience.py`：韧性指标与熵权法、AR(1)。
- `model.py`：XGBoost 与 SHAP 分析。
- `run_pipeline.py`：端到端执行脚本。

---

## 依赖安装

```bash
pip install xarray numpy pandas matplotlib xgboost shap scikit-learn netCDF4 dask
```

---

## 运行示例

```bash
python run_pipeline.py \
  --nc_file "C:/Users/Administrator/outputs/Yunnan_EcoHydrology_Final.nc" \
  --outdir outputs \
  --resolution 0.05 \
  --gpp_var gpp_anom \
  --drought_var spei_anom
```

> 论文原文使用 0.1°，这里通过 `--resolution 0.05` 按你的要求运行。

---

## 输入数据要求

至少需要变量：

- `time`, `lat`, `lon`
- `gpp_anom`（可替换为你自己的 GPP 字段）
- `spei_anom`（作为 MCDI 代理，可替换为你的 MCDI 字段）

可选驱动变量（用于 XGBoost）：

- `sm_anom`, `t2m_anom`, `rad_anom`, `evi_anom`, `gosif_anom`
- 静态变量建议先拼接到事件表：`dem`, `awc`, `cec` 等

可选生态系统类别：

- 事件表中的 `eco_type`（forest/grassland/cropland）用于时间序列分组绘图。

---

## 输出结果

脚本会在 `outdir` 输出：

- `drought_events.csv`：每个像元干旱事件起止；
- `event_metrics_resilience.csv`：事件级所有指标；
- `entropy_weights.csv`：熵权法权重；
- `resilience_maps.nc`：空间指标栅格（R/C/A/Resilience）；
- `map_*.png`：空间分布图；
- `ar1_series.nc`、`ar1_timeseries.png`：AR(1) 验证；
- `cv_pred_*.csv`、`xgboost_cv_r2.csv`：建模结果；
- `shap_summary_*.png`、`shap_dependence_*.png`、`shap_importance_*.csv`：SHAP解释。

---

## 方法与论文对应关系（简要）

- **Step1**：`preprocess.py` 中 `deseasonalize_monthly` + `detrend_per_pixel` + `gpp_neg_mask`；
- **Step2**：`drought_detection.py` 中 `_find_events_1d`；
- **Step3-4**：`resilience.py` 中 `compute_event_metrics`；
- **Step5**：`entropy_weights` + `compute_resilience`；
- **Step6**：`rolling_ar1`；
- **Step7**：`model.py` 中 `train_xgb_cv` + `run_shap`。

---

## 注意事项

1. 如果你的干旱指数不是 `spei_anom`，请改 `--drought_var`。
2. 如果你已有“去季节 + 去趋势”的 GPP，可在预处理中跳过对应步骤。
3. 大规模数据建议用 dask chunk 并按区域分块运行。
4. Adaptability 对首个事件无法计算 `Aprev`，会出现 NaN，这是方法定义导致的正常现象。
