from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR


def _demean(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values - values.groupby(groups).transform("mean")


def _fit_single_regressor(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    denominator = float(x @ x)
    if denominator <= 1e-12:
        raise ValueError("Regressor has no within-group variation")
    coefficient = float((x @ y) / denominator)
    residual = y - coefficient * x
    sigma2 = float((residual @ residual) / max(len(x) - 1, 1))
    standard_error = float(np.sqrt(sigma2 / denominator))
    return coefficient, standard_error


def run_elasticity_baseline(category: str, holdout_weeks: int = 26) -> tuple[Path, Path]:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}; run the build command first")
    frame = pd.read_csv(
        panel_path,
        usecols=["upc", "store", "week", "move", "unit_price", "log_unit_price"],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )

    # Restrict the diagnostic baseline to positive sales and sufficiently observed panels.
    frame = frame.loc[(frame["move"] > 0) & np.isfinite(frame["log_unit_price"])].copy()
    frame["log_units"] = np.log(frame["move"])
    frame["panel_id"] = frame["store"].astype(str) + ":" + frame["upc"].astype(str)
    group_counts = frame.groupby("panel_id")["week"].transform("count")
    price_counts = frame.groupby("panel_id")["unit_price"].transform("nunique")
    frame = frame.loc[(group_counts >= 40) & (price_counts >= 3)].copy()
    if frame.empty:
        raise ValueError("No panels satisfy the baseline support requirements")

    max_week = int(frame["week"].max())
    cutoff = max_week - holdout_weeks
    train = frame.loc[frame["week"] <= cutoff].copy()
    test = frame.loc[frame["week"] > cutoff].copy()

    train["x_within"] = _demean(train["log_unit_price"], train["panel_id"])
    train["y_within"] = _demean(train["log_units"], train["panel_id"])
    coefficient, standard_error = _fit_single_regressor(
        train["x_within"].to_numpy(), train["y_within"].to_numpy()
    )

    panel_means = train.groupby("panel_id")[["log_unit_price", "log_units"]].mean()
    test = test.join(panel_means, on="panel_id", rsuffix="_train_mean")
    supported = test["log_units_train_mean"].notna()
    test = test.loc[supported].copy()
    test["prediction"] = test["log_units_train_mean"] + coefficient * (
        test["log_unit_price"] - test["log_unit_price_train_mean"]
    )
    mae = float(np.mean(np.abs(test["log_units"] - test["prediction"]))) if len(test) else None
    rmse = (
        float(np.sqrt(np.mean((test["log_units"] - test["prediction"]) ** 2)))
        if len(test)
        else None
    )

    result = {
        "category": category,
        "model": "store-product fixed-effects log-log diagnostic baseline",
        "interpretation": "Within-panel association; not a causal price elasticity estimate.",
        "coefficient": coefficient,
        "standard_error_naive": standard_error,
        "approx_95pct_interval_naive": [
            coefficient - 1.96 * standard_error,
            coefficient + 1.96 * standard_error,
        ],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "panels": int(train["panel_id"].nunique()),
        "train_max_week": cutoff,
        "test_max_week": max_week,
        "holdout_log_mae": mae,
        "holdout_log_rmse": rmse,
    }
    result_path = ARTIFACT_DIR / f"{category}_baseline.json"
    predictions_path = ARTIFACT_DIR / f"{category}_baseline_holdout.csv.gz"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test[["upc", "store", "week", "unit_price", "move", "log_units", "prediction"]].to_csv(
        predictions_path, index=False, compression="gzip"
    )
    return result_path, predictions_path
