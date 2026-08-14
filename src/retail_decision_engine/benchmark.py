from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR


def _two_way_residualize(
    frame: pd.DataFrame,
    columns: list[str],
    first_effect: str,
    second_effect: str,
    iterations: int = 6,
) -> pd.DataFrame:
    residual = frame[columns].astype("float64").copy()
    for _ in range(iterations):
        residual -= residual.groupby(frame[first_effect], sort=False).transform("mean")
        residual -= residual.groupby(frame[second_effect], sort=False).transform("mean")
    return residual


def _clustered_ols(
    x: np.ndarray, y: np.ndarray, clusters: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    bread = np.linalg.inv(x.T @ x)

    scores = pd.DataFrame(x * residuals[:, None])
    scores["cluster"] = clusters.to_numpy()
    cluster_sums = scores.groupby("cluster", sort=False).sum().to_numpy()
    meat = cluster_sums.T @ cluster_sums
    covariance = bread @ meat @ bread

    n, k = x.shape
    groups = len(cluster_sums)
    if groups > 1 and n > k:
        covariance *= (groups / (groups - 1)) * ((n - 1) / (n - k))
    return coefficients, np.sqrt(np.diag(covariance))


def run_two_way_benchmark(category: str, holdout_weeks: int = 26) -> Path:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}; run the build command first")

    frame = pd.read_csv(
        panel_path,
        usecols=["upc", "store", "week", "move", "unit_price", "sale"],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )
    frame = frame.loc[(frame["move"] > 0) & (frame["unit_price"] > 0)].copy()
    frame["panel_id"] = frame["store"].astype(str) + ":" + frame["upc"].astype(str)

    counts = frame.groupby("panel_id")["week"].transform("count")
    price_levels = frame.groupby("panel_id")["unit_price"].transform("nunique")
    median_price = frame.groupby("panel_id")["unit_price"].transform("median")
    relative_price = frame["unit_price"] / median_price
    frame = frame.loc[
        (counts >= 40) & (price_levels >= 3) & relative_price.between(0.25, 4.0)
    ].copy()

    cutoff = int(frame["week"].max()) - holdout_weeks
    train = frame.loc[frame["week"] <= cutoff].copy()
    train["log_units"] = np.log(train["move"])
    train["log_price"] = np.log(train["unit_price"])

    sale = train["sale"].fillna("").astype(str).str.strip().str.upper()
    for code in ("B", "S", "C", "G"):
        train[f"promo_{code}"] = sale.eq(code).astype("float64")

    regressors = ["log_price", "promo_B", "promo_S", "promo_C", "promo_G"]
    residualized = _two_way_residualize(
        train,
        ["log_units", *regressors],
        first_effect="panel_id",
        second_effect="week",
    )
    x = residualized[regressors].to_numpy()
    y = residualized["log_units"].to_numpy()
    coefficients, standard_errors = _clustered_ols(x, y, train["panel_id"])

    estimates: dict[str, dict[str, float]] = {}
    for name, coefficient, standard_error in zip(regressors, coefficients, standard_errors):
        estimate = {
            "coefficient": float(coefficient),
            "clustered_standard_error": float(standard_error),
            "approx_95pct_low": float(coefficient - 1.96 * standard_error),
            "approx_95pct_high": float(coefficient + 1.96 * standard_error),
        }
        if name.startswith("promo_"):
            estimate["conditional_unit_difference_pct"] = float(100 * np.expm1(coefficient))
        estimates[name] = estimate

    result = {
        "category": category,
        "model": "store-product and week fixed effects with observed promotion-code controls",
        "research_question": (
            "Within the same store-product panel, how are units associated with price changes, "
            "and do recorded promotion types carry additional demand signal beyond price?"
        ),
        "causal_status": (
            "Associational benchmark only: price/promotion assignment can be endogenous and "
            "the source manual states promotion flags are incomplete."
        ),
        "price_filter": "0.25x to 4x each panel's median observed unit price",
        "train_rows": int(len(train)),
        "panels": int(train["panel_id"].nunique()),
        "weeks": int(train["week"].nunique()),
        "train_max_week": cutoff,
        "estimates": estimates,
    }
    output = ARTIFACT_DIR / f"{category}_two_way_benchmark.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
