from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Quantile-bin PSI with stable handling of repeated values and empty bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("reference and current must contain finite observations")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(reference.mean(), current.mean()) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    epsilon = 1e-6
    reference_share = np.maximum(reference_counts / reference_counts.sum(), epsilon)
    current_share = np.maximum(current_counts / current_counts.sum(), epsilon)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def drift_status(psi: float, warning: float = 0.10, block: float = 0.25) -> str:
    if psi >= block:
        return "block"
    if psi >= warning:
        return "warn"
    return "pass"


def _metric_status(value: float, pass_threshold: float, block_threshold: float) -> str:
    if value >= pass_threshold:
        return "pass"
    if value >= block_threshold:
        return "warn"
    return "block"


def run_historical_shadow_replay(
    category: str,
    lookback_weeks: int = 13,
    coverage_target: float = 0.90,
    coverage_block: float = 0.85,
) -> tuple[Path, Path]:
    """Replay an untouched holdout sequentially using strictly prior calibration errors."""
    scored_path = ARTIFACT_DIR / f"{category}_predictive_holdout.csv.gz"
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing {scored_path}; run calibration first")
    scored = pd.read_csv(scored_path)
    panel = pd.read_csv(
        PROCESSED_DIR / f"{category}_store_product_week.csv.gz",
        usecols=["upc", "store", "week", "log_unit_price", "promotion_observed"],
        low_memory=False,
    )
    replay = scored.merge(
        panel,
        on=["upc", "store", "week"],
        how="left",
        validate="one_to_one",
    )
    replay["absolute_error"] = (
        replay["log_units"] - replay["predictive_p50"]
    ).abs()
    weeks = np.sort(replay["week"].unique())
    split_week = int(weeks[len(weeks) // 2])
    replay_weeks = weeks[weeks >= split_week]
    rows = []
    scored_parts = []
    for week in replay_weeks:
        history = replay.loc[
            replay["week"].between(week - lookback_weeks, week - 1)
        ].copy()
        current = replay.loc[replay["week"].eq(week)].copy()
        n = len(history)
        level = min(1.0, np.ceil((n + 1) * coverage_target) / n)
        radius = float(np.quantile(history["absolute_error"], level, method="higher"))
        current["rolling_conformal_low"] = current["predictive_p50"] - radius
        current["rolling_conformal_high"] = current["predictive_p50"] + radius
        current["rolling_conformal_covered"] = current["log_units"].between(
            current["rolling_conformal_low"], current["rolling_conformal_high"]
        )
        coverage = float(current["rolling_conformal_covered"].mean())
        price_psi = population_stability_index(
            history["log_unit_price"].to_numpy(), current["log_unit_price"].to_numpy()
        )
        residual_bias = float(
            (current["log_units"] - current["predicted_log_units"]).mean()
        )
        statuses = {
            "coverage": _metric_status(coverage, coverage_target, coverage_block),
            "price_psi": drift_status(price_psi),
            "absolute_bias": "pass"
            if abs(residual_bias) <= 0.10
            else "warn"
            if abs(residual_bias) <= 0.20
            else "block",
        }
        overall = (
            "block"
            if "block" in statuses.values()
            else "warn"
            if "warn" in statuses.values()
            else "pass"
        )
        rows.append(
            {
                "week": int(week),
                "rows": int(len(current)),
                "history_rows": int(n),
                "rolling_radius_log_units": radius,
                "rolling_90_coverage": coverage,
                "log_mae": float(current["absolute_error"].mean()),
                "mean_bias_actual_minus_predicted": residual_bias,
                "price_psi": price_psi,
                "promotion_rate": float(current["promotion_observed"].mean()),
                "coverage_status": statuses["coverage"],
                "price_psi_status": statuses["price_psi"],
                "bias_status": statuses["absolute_bias"],
                "batch_status": overall,
            }
        )
        scored_parts.append(current)

    batches = pd.DataFrame(rows)
    replay_scored = pd.concat(scored_parts, ignore_index=True)
    aggregate_coverage = float(replay_scored["rolling_conformal_covered"].mean())
    aggregate_status = _metric_status(
        aggregate_coverage, coverage_target, coverage_block
    )
    overall_status = (
        "block"
        if batches["batch_status"].eq("block").any()
        else "warn"
        if batches["batch_status"].eq("warn").any()
        else "pass"
    )
    highest_risk = batches.sort_values(
        ["price_psi", "rolling_90_coverage"], ascending=[False, True]
    ).iloc[0]
    summary = {
        "category": category,
        "status": "historical_shadow_replay_complete",
        "evidence_scope": (
            "Sequential replay of the untouched temporal holdout. This tests monitoring and "
            "past-only recalibration behavior; it is not live production evidence."
        ),
        "replay_rows": int(len(replay_scored)),
        "replay_weeks": int(len(replay_weeks)),
        "start_week": int(replay_weeks.min()),
        "end_week": int(replay_weeks.max()),
        "lookback_weeks": lookback_weeks,
        "rolling_90_coverage": aggregate_coverage,
        "coverage_target": coverage_target,
        "coverage_status": aggregate_status,
        "overall_status": overall_status,
        "mean_log_mae": float(replay_scored["absolute_error"].mean()),
        "mean_bias_actual_minus_predicted": float(
            (replay_scored["log_units"] - replay_scored["predicted_log_units"]).mean()
        ),
        "weeks_by_status": {
            status: int(batches["batch_status"].eq(status).sum())
            for status in ("pass", "warn", "block")
        },
        "maximum_price_psi": float(batches["price_psi"].max()),
        "highest_risk_week": {
            "week": int(highest_risk["week"]),
            "price_psi": float(highest_risk["price_psi"]),
            "promotion_rate": float(highest_risk["promotion_rate"]),
            "rolling_90_coverage": float(highest_risk["rolling_90_coverage"]),
            "mean_bias_actual_minus_predicted": float(
                highest_risk["mean_bias_actual_minus_predicted"]
            ),
        },
        "monitoring_fields": [
            "rolling predictive coverage",
            "log-scale error and bias",
            "price population stability",
            "promotion-rate context",
        ],
        "decision": (
            "Monitoring machinery is exercised on real historical batches. Coverage remains "
            "below target, so the replay supports continued human review rather than automated use."
        ),
    }
    summary_path = ARTIFACT_DIR / f"{category}_historical_shadow_replay.json"
    batches_path = ARTIFACT_DIR / f"{category}_historical_shadow_batches.csv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    batches.to_csv(batches_path, index=False)
    return summary_path, batches_path
