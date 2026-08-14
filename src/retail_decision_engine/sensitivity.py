from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .bayesian import run_hierarchical_model
from .config import ARTIFACT_DIR


def run_prior_sensitivity(
    category: str,
    n_products: int = 8,
    n_stores: int = 12,
    history_weeks: int = 156,
    holdout_weeks: int = 26,
    draws: int = 300,
    tune: int = 300,
) -> Path:
    for profile in ("skeptical", "weak"):
        run_hierarchical_model(
            category,
            n_products=n_products,
            n_stores=n_stores,
            history_weeks=history_weeks,
            holdout_weeks=holdout_weeks,
            draws=draws,
            tune=tune,
            cores=1,
            prior_profile=profile,
        )

    profiles = {}
    product_frames = []
    for profile, suffix in (
        ("regularized", ""),
        ("skeptical", "_skeptical"),
        ("weak", "_weak"),
    ):
        summary_path = ARTIFACT_DIR / f"{category}_hierarchical{suffix}_summary.json"
        elasticity_path = ARTIFACT_DIR / f"{category}_product_elasticities{suffix}.csv"
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        profiles[profile] = {
            "mean_elasticity": summary["mean_elasticity"],
            "holdout_log_mae": summary["holdout_log_mae"],
            "holdout_log_rmse": summary["holdout_log_rmse"],
            "diagnostics": summary["diagnostics"],
        }
        frame = pd.read_csv(elasticity_path)
        product_frames.append(
            frame[["upc", "description", "posterior_mean"]].rename(
                columns={"posterior_mean": profile}
            )
        )

    comparison = product_frames[0]
    for frame in product_frames[1:]:
        comparison = comparison.merge(frame, on=["upc", "description"], how="outer")
    comparison["range_across_priors"] = comparison[["regularized", "skeptical", "weak"]].max(
        axis=1
    ) - comparison[["regularized", "skeptical", "weak"]].min(axis=1)
    comparison_path = ARTIFACT_DIR / f"{category}_prior_sensitivity_by_product.csv"
    comparison.to_csv(comparison_path, index=False)

    mean_values = [profiles[name]["mean_elasticity"]["posterior_mean"] for name in profiles]
    result = {
        "category": category,
        "profiles": profiles,
        "global_mean_range_across_priors": float(max(mean_values) - min(mean_values)),
        "maximum_product_mean_range_across_priors": float(comparison["range_across_priors"].max()),
        "robustness_rule": (
            "Global range below 0.25 and product range below 0.50 are treated as practically stable "
            "for this screening analysis; convergence diagnostics still govern usability."
        ),
        "comparison_file": str(comparison_path),
    }
    output = ARTIFACT_DIR / f"{category}_prior_sensitivity.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
