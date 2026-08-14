from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy import stats

from .bayesian import _linear_predictor_draws, _posterior_samples, _prepare_cohort
from .config import ARTIFACT_DIR


def _coverage(
    actual: np.ndarray,
    draws: np.ndarray,
    probability: float,
    clusters: np.ndarray | None = None,
    seed: int = 20260720,
) -> dict[str, float]:
    tail = (1 - probability) / 2
    low = np.quantile(draws, tail, axis=1)
    high = np.quantile(draws, 1 - tail, axis=1)
    covered = (actual >= low) & (actual <= high)
    result = {
        "nominal": probability,
        "empirical": float(covered.mean()),
        "mean_width": float(np.mean(high - low)),
    }
    if clusters is not None:
        codes, levels = pd.factorize(clusters, sort=True)
        cluster_covered = np.bincount(codes, weights=covered.astype(float))
        cluster_rows = np.bincount(codes)
        rng = np.random.default_rng(seed + int(probability * 100))
        selections = rng.integers(0, len(levels), size=(1000, len(levels)))
        bootstrap = cluster_covered[selections].sum(axis=1) / cluster_rows[selections].sum(axis=1)
        result["cluster_bootstrap_low"] = float(np.quantile(bootstrap, 0.025))
        result["cluster_bootstrap_high"] = float(np.quantile(bootstrap, 0.975))
        result["cluster_count"] = int(len(levels))
    return result


def _split_conformal_temporal(
    actual: np.ndarray,
    center: np.ndarray,
    weeks: np.ndarray,
    clusters: np.ndarray,
    probabilities: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
    seed: int = 20260720,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Calibrate symmetric intervals on early weeks and evaluate on later weeks.

    The chronological split avoids evaluating on observations used to choose the
    conformal radius. Coverage is empirical because temporal exchangeability is
    not assumed to hold for this retail panel.
    """
    unique_weeks = np.sort(np.unique(weeks))
    split_week = unique_weeks[len(unique_weeks) // 2]
    calibration = weeks < split_week
    evaluation = ~calibration
    scores = np.abs(actual[calibration] - center[calibration])
    results: dict[str, object] = {}
    lower_90 = np.full(len(actual), np.nan)
    upper_90 = np.full(len(actual), np.nan)
    for probability in probabilities:
        finite_sample_level = min(
            1.0,
            np.ceil((len(scores) + 1) * probability) / len(scores),
        )
        radius = float(np.quantile(scores, finite_sample_level, method="higher"))
        low = center[evaluation] - radius
        high = center[evaluation] + radius
        covered = (actual[evaluation] >= low) & (actual[evaluation] <= high)
        evaluation_clusters = clusters[evaluation]
        codes, levels = pd.factorize(evaluation_clusters, sort=True)
        cluster_covered = np.bincount(codes, weights=covered.astype(float))
        cluster_rows = np.bincount(codes)
        rng = np.random.default_rng(seed + int(probability * 1000))
        selections = rng.integers(0, len(levels), size=(1000, len(levels)))
        bootstrap = cluster_covered[selections].sum(axis=1) / cluster_rows[selections].sum(axis=1)
        results[str(int(probability * 100))] = {
            "nominal": probability,
            "radius_log_units": radius,
            "evaluation_coverage": float(covered.mean()),
            "cluster_bootstrap_low": float(np.quantile(bootstrap, 0.025)),
            "cluster_bootstrap_high": float(np.quantile(bootstrap, 0.975)),
        }
        if probability == 0.90:
            lower_90[evaluation] = low
            upper_90[evaluation] = high
    metadata = {
        "method": "chronological split conformal around the posterior predictive median",
        "calibration_rows": int(calibration.sum()),
        "evaluation_rows": int(evaluation.sum()),
        "split_week": int(split_week),
        "intervals": results,
        "interpretation": (
            "Post-hoc uncertainty repair evaluated strictly after its calibration window. "
            "Finite-sample conformal guarantees require exchangeability, which is not assumed "
            "for this time-dependent panel; observed evaluation coverage remains the gate."
        ),
    }
    return metadata, lower_90, upper_90


def run_posterior_predictive_calibration(
    category: str,
    n_products: int = 8,
    n_stores: int = 12,
    history_weeks: int = 156,
    holdout_weeks: int = 26,
    seed: int = 20260720,
) -> tuple[Path, Path, Path]:
    posterior_path = ARTIFACT_DIR / f"{category}_hierarchical_posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing {posterior_path}; run the Bayesian model first")
    idata = az.from_netcdf(posterior_path)
    _, holdout, products, _, product_names = _prepare_cohort(
        category, n_products, n_stores, history_weeks, holdout_weeks
    )

    linear_draws = _linear_predictor_draws(idata, holdout)
    sigma = _posterior_samples(idata, "sigma")
    nu = _posterior_samples(idata, "nu_minus_two") + 2
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(df=nu[None, :], size=linear_draws.shape) * sigma[None, :]
    predictive_draws = linear_draws + noise
    actual = holdout["log_units"].to_numpy()
    predictive_mean = predictive_draws.mean(axis=1)
    residual = actual - predictive_mean
    pit = (predictive_draws <= actual[:, None]).mean(axis=1)
    ks = stats.kstest(pit, "uniform")

    holdout_clusters = (holdout["upc"].astype(str) + ":" + holdout["store"].astype(str)).to_numpy()
    interval_results = {
        str(int(probability * 100)): _coverage(
            actual, predictive_draws, probability, clusters=holdout_clusters, seed=seed
        )
        for probability in (0.50, 0.80, 0.90, 0.95)
    }
    conformal, conformal_low_90, conformal_high_90 = _split_conformal_temporal(
        actual,
        np.quantile(predictive_draws, 0.50, axis=1),
        holdout["week"].to_numpy(),
        holdout_clusters,
        seed=seed,
    )
    predictive_means_by_draw = predictive_draws.mean(axis=0)
    predictive_stds_by_draw = predictive_draws.std(axis=0)
    summary = {
        "category": category,
        "calibration_target": (
            "conditional temporal holdout scoring using observed holdout prices and promotion codes; "
            "not an advance forecast of unknown future covariates"
        ),
        "holdout_rows": int(len(holdout)),
        "posterior_draws": int(predictive_draws.shape[1]),
        "log_scale_error": {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mean_bias_actual_minus_predicted": float(residual.mean()),
        },
        "predictive_interval_calibration": interval_results,
        "temporal_split_conformal_recalibration": conformal,
        "pit_uniformity": {
            "mean": float(pit.mean()),
            "std": float(pit.std()),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "interpretation": (
                "Reference-only iid KS diagnostic. Store-product-week dependence makes its p-value "
                "anti-conservative; deployment decisions use coverage and cluster-bootstrap intervals."
            ),
        },
        "posterior_predictive_checks": {
            "observed_log_mean": float(actual.mean()),
            "predictive_log_mean_p05": float(np.quantile(predictive_means_by_draw, 0.05)),
            "predictive_log_mean_p95": float(np.quantile(predictive_means_by_draw, 0.95)),
            "observed_log_std": float(actual.std()),
            "predictive_log_std_p05": float(np.quantile(predictive_stds_by_draw, 0.05)),
            "predictive_log_std_p95": float(np.quantile(predictive_stds_by_draw, 0.95)),
        },
    }
    summary_path = ARTIFACT_DIR / f"{category}_predictive_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scored = holdout[["upc", "store", "week", "move", "log_units"]].copy()
    scored["predicted_log_units"] = predictive_mean
    scored["predictive_p05"] = np.quantile(predictive_draws, 0.05, axis=1)
    scored["predictive_p50"] = np.quantile(predictive_draws, 0.50, axis=1)
    scored["predictive_p95"] = np.quantile(predictive_draws, 0.95, axis=1)
    scored["pit"] = pit
    scored["conformal_p05"] = conformal_low_90
    scored["conformal_p95"] = conformal_high_90
    scored_path = ARTIFACT_DIR / f"{category}_predictive_holdout.csv.gz"
    scored.to_csv(scored_path, index=False, compression="gzip")

    product_rows = []
    for product_index, upc in enumerate(products):
        mask = holdout["product_idx"].eq(product_index).to_numpy()
        product_actual = actual[mask]
        product_prediction = predictive_mean[mask]
        product_draws = predictive_draws[mask]
        low = np.quantile(product_draws, 0.05, axis=1)
        high = np.quantile(product_draws, 0.95, axis=1)
        product_rows.append(
            {
                "upc": upc,
                "description": product_names.get(upc, ""),
                "rows": int(mask.sum()),
                "log_mae": float(np.mean(np.abs(product_actual - product_prediction))),
                "log_bias_actual_minus_predicted": float(
                    np.mean(product_actual - product_prediction)
                ),
                "coverage_90": float(((product_actual >= low) & (product_actual <= high)).mean()),
                "mean_pit": float(pit[mask].mean()),
            }
        )
    product_path = ARTIFACT_DIR / f"{category}_predictive_calibration_by_product.csv"
    pd.DataFrame(product_rows).to_csv(product_path, index=False)
    return summary_path, scored_path, product_path
