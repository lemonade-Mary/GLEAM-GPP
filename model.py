"""XGBoost + SHAP driver analysis for resilience metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


TARGETS = ["resistance", "recovery", "adaptability", "resilience"]


def prepare_model_table(
    events_df: pd.DataFrame,
    dynamic_features: Iterable[str],
    static_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build model-ready feature table.

    events_df should already include drought features (duration, severity, etc.) and
    target metrics.
    """
    cols = ["lat", "lon", "start_idx", "end_idx", "duration", *dynamic_features, *TARGETS]
    cols = [c for c in cols if c in events_df.columns]
    tbl = events_df[cols].copy()

    if static_df is not None:
        tbl = tbl.merge(static_df, on=["lat", "lon"], how="left")

    tbl = tbl.replace([np.inf, -np.inf], np.nan).dropna()
    return tbl


def train_xgb_cv(
    table: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    group_cols: Tuple[str, str] = ("lat", "lon"),
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[xgb.XGBRegressor, float, np.ndarray, pd.DataFrame]:
    """Train grouped CV XGBoost model and return final model + out-of-fold predictions."""
    X = table[feature_cols]
    y = table[target].values
    groups = table[list(group_cols)].astype(str).agg("_".join, axis=1)

    cv = GroupKFold(n_splits=n_splits)
    oof = np.full(len(table), np.nan)

    params = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=4,
    )

    for tr, te in cv.split(X, y, groups=groups):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])

    score = r2_score(y[~np.isnan(oof)], oof[~np.isnan(oof)])

    final_model = xgb.XGBRegressor(**params)
    final_model.fit(X, y)

    pred_df = table[[*group_cols]].copy()
    pred_df["y_true"] = y
    pred_df["y_pred"] = oof
    return final_model, score, oof, pred_df


def run_shap(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    outdir: str,
    target_name: str,
    top_n: int = 10,
) -> Dict[str, str]:
    """Generate SHAP summary and dependence plot paths."""
    out = {}
    Path(outdir).mkdir(parents=True, exist_ok=True)

    explainer = shap.Explainer(model)
    sv = explainer(X)

    summary_path = str(Path(outdir) / f"shap_summary_{target_name}.png")
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(sv, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200)
    plt.close()
    out["summary"] = summary_path

    # Dependence plot for top feature by mean|SHAP|.
    mean_abs = np.abs(sv.values).mean(axis=0)
    top_feat = X.columns[int(np.argmax(mean_abs))]
    dep_path = str(Path(outdir) / f"shap_dependence_{target_name}_{top_feat}.png")
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(top_feat, sv.values, X, show=False)
    plt.tight_layout()
    plt.savefig(dep_path, dpi=200)
    plt.close()
    out["dependence"] = dep_path
    out["top_feature"] = top_feat

    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    imp_path = str(Path(outdir) / f"shap_importance_{target_name}.csv")
    imp.to_csv(imp_path, index=False)
    out["importance_csv"] = imp_path

    return out
