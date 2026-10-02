from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import wilson_interval, bootstrap_balanced_accuracy


def selection_rate_summary(df: pd.DataFrame, group_cols: list[str], selected_col: str = "bayes_selected") -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols + ["true_model"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        *cond, true_model = keys
        success = int((g[selected_col] == "M2").sum())
        n = len(g)
        lo, hi = wilson_interval(success, n)
        row = {c: v for c, v in zip(group_cols, cond)}
        row.update({
            "true_model": true_model,
            "rows": n,
            "p_select_M2": success / n,
            "p_select_M2_ci_low": lo,
            "p_select_M2_ci_high": hi,
            "median_logBF": float(g["logBF"].median()) if "logBF" in g else np.nan,
            "q1_logBF": float(g["logBF"].quantile(0.25)) if "logBF" in g else np.nan,
            "q3_logBF": float(g["logBF"].quantile(0.75)) if "logBF" in g else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def balanced_accuracy_summary(df: pd.DataFrame, group_cols: list[str], selected_col: str = "bayes_selected", seed: int = 1234) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        point, lo, hi = bootstrap_balanced_accuracy(g["true_model"], g[selected_col], seed=seed)
        m1 = g[g["true_model"] == "M1"]
        m2 = g[g["true_model"] == "M2"]
        fpr = float(np.mean(m1[selected_col] == "M2")) if len(m1) else np.nan
        tpr = float(np.mean(m2[selected_col] == "M2")) if len(m2) else np.nan
        row = {c: v for c, v in zip(group_cols, keys)}
        row.update({
            "TPR": tpr,
            "FPR": fpr,
            "balanced_accuracy": point,
            "balanced_accuracy_ci_low": lo,
            "balanced_accuracy_ci_high": hi,
            "rows": len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)
