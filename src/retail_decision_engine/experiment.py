from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR


EXPERIMENT_REQUIRED_COLUMNS = (
    "experiment_id",
    "category",
    "store",
    "upc",
    "week",
    "randomization_unit_id",
    "randomization_block",
    "assignment_source",
    "randomization_seed",
    "assigned_arm",
    "assigned_treatment",
    "assignment_probability",
    "explicit_discount_depth",
    "display_flag",
    "feature_flag",
    "offer_delivered",
    "regular_price",
    "offered_price",
    "units_sold",
    "replacement_unit_cost",
    "supplier_funding_per_unit",
    "inventory_on_hand_start",
    "stockout_flag",
    "baseline_units_8w",
    "baseline_margin_8w",
    "basket_margin",
    "category_substitution_units",
)


def _boolean_series(series: pd.Series) -> pd.Series:
    mapped = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"true", "1", "yes"}
        if str(value).strip().lower() in {"true", "false", "1", "0", "yes", "no"}
        else np.nan
    )
    return mapped.astype("boolean")


def validate_experiment_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate whether a randomized promotion panel can support an ITT analysis."""
    missing = sorted(set(EXPERIMENT_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        return {
            "status": "blocked_missing_columns",
            "rows": int(len(frame)),
            "missing_columns": missing,
            "checks": {"required_columns_present": False},
            "intake_passed": False,
        }

    work = frame.loc[:, EXPERIMENT_REQUIRED_COLUMNS].copy()
    numeric_columns = (
        "store",
        "week",
        "randomization_seed",
        "assigned_treatment",
        "assignment_probability",
        "explicit_discount_depth",
        "regular_price",
        "offered_price",
        "units_sold",
        "replacement_unit_cost",
        "supplier_funding_per_unit",
        "inventory_on_hand_start",
        "baseline_units_8w",
        "baseline_margin_8w",
        "basket_margin",
        "category_substitution_units",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in ("display_flag", "feature_flag", "offer_delivered", "stockout_flag"):
        work[column] = _boolean_series(work[column])

    treated = work["assigned_treatment"].eq(1)
    control = work["assigned_treatment"].eq(0)
    has_rows = len(work) > 0
    key = ["experiment_id", "store", "upc", "week"]
    economic_fields = ["replacement_unit_cost", "supplier_funding_per_unit"]
    mechanics_fields = [
        "explicit_discount_depth",
        "display_flag",
        "feature_flag",
        "regular_price",
        "offered_price",
    ]
    checks = {
        "required_columns_present": True,
        "rows_present": has_rows,
        "unique_store_item_week_assignment": has_rows and not work.duplicated(key).any(),
        "randomized_assignment_declared": has_rows
        and work["assignment_source"]
        .astype(str)
        .str.lower()
        .eq("randomized")
        .all(),
        "randomization_metadata_complete": has_rows
        and work[
            ["randomization_unit_id", "randomization_block", "randomization_seed"]
        ]
        .notna()
        .all()
        .all(),
        "control_and_treatment_present": bool(control.any() and treated.any()),
        "assignment_probability_valid": has_rows
        and work["assignment_probability"].between(0, 1, inclusive="neither").all(),
        "offer_mechanics_complete": bool(treated.any())
        and work.loc[treated, mechanics_fields].notna().all().all(),
        "treated_discount_positive": bool(treated.any())
        and work.loc[treated, "explicit_discount_depth"].gt(0).all(),
        "control_discount_zero": bool(control.any())
        and work.loc[control, "explicit_discount_depth"].eq(0).all(),
        "offered_price_valid": has_rows
        and (
            work["offered_price"].gt(0)
            & work["regular_price"].gt(0)
            & work["offered_price"].le(work["regular_price"])
        ).all(),
        "zero_sales_capable_panel": has_rows
        and work["units_sold"].notna().all()
        and work["units_sold"].ge(0).all()
        and work.loc[work["units_sold"].eq(0), "offered_price"].notna().all(),
        "inventory_and_stockout_complete": has_rows
        and work[
            ["inventory_on_hand_start", "stockout_flag"]
        ]
        .notna()
        .all()
        .all()
        and work["inventory_on_hand_start"].ge(0).all(),
        "replacement_cost_and_funding_complete": has_rows
        and work[economic_fields].notna().all().all()
        and work[economic_fields].ge(0).all().all(),
        "baseline_covariates_complete": has_rows
        and work[
            ["baseline_units_8w", "baseline_margin_8w"]
        ]
        .notna()
        .all()
        .all(),
        "business_guardrails_complete": has_rows
        and work[
            ["basket_margin", "category_substitution_units"]
        ]
        .notna()
        .all()
        .all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    zero_rows = int(work["units_sold"].eq(0).sum())
    passed = all(checks.values())
    return {
        "status": "ready_for_randomized_itt_analysis"
        if passed
        else "awaiting_experiment_data"
        if not has_rows
        else "blocked_data_quality",
        "rows": int(len(work)),
        "experiments": int(work["experiment_id"].nunique()),
        "arms": sorted(work["assigned_arm"].dropna().astype(str).unique().tolist()),
        "stores": int(work["store"].nunique()),
        "zero_sales_rows": zero_rows,
        "zero_sales_share": float(zero_rows / len(work)) if len(work) else None,
        "missing_columns": [],
        "checks": checks,
        "intake_passed": passed,
        "provenance_note": (
            "The validator confirms declared randomization metadata and analytic completeness; "
            "it cannot independently prove that assignment was executed as declared. Preserve the "
            "allocation file and experiment registration for audit."
        ),
    }


def _clustered_ols(
    outcome: np.ndarray, design: np.ndarray, clusters: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    xtx_inverse = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residual = outcome - design @ coefficients
    meat = np.zeros((design.shape[1], design.shape[1]))
    cluster_values = clusters.astype(str).to_numpy()
    levels = np.unique(cluster_values)
    for level in levels:
        mask = cluster_values == level
        score = design[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = xtx_inverse @ meat @ xtx_inverse
    n, parameters = design.shape
    if len(levels) > 1 and n > parameters:
        covariance *= len(levels) / (len(levels) - 1) * (n - 1) / (n - parameters)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    return coefficients, standard_errors


def _itt_by_arm(frame: pd.DataFrame, outcome: str) -> list[dict[str, Any]]:
    arms = sorted(frame.loc[frame["assigned_treatment"].eq(1), "assigned_arm"].unique())
    blocks = pd.get_dummies(frame["randomization_block"].astype(str), drop_first=True, dtype=float)
    treatment = pd.DataFrame(
        {f"arm::{arm}": frame["assigned_arm"].eq(arm).astype(float) for arm in arms}
    )
    design_frame = pd.concat(
        [pd.Series(1.0, index=frame.index, name="intercept"), treatment, blocks], axis=1
    )
    design = design_frame.to_numpy(dtype=float)
    coefficients, standard_errors = _clustered_ols(
        frame[outcome].to_numpy(dtype=float), design, frame["store"]
    )
    results = []
    for arm in arms:
        index = design_frame.columns.get_loc(f"arm::{arm}")
        estimate = float(coefficients[index])
        standard_error = float(standard_errors[index])
        results.append(
            {
                "arm": str(arm),
                "estimate": estimate,
                "clustered_standard_error": standard_error,
                "ci95_low": estimate - 1.96 * standard_error,
                "ci95_high": estimate + 1.96 * standard_error,
            }
        )
    return results


def analyze_randomized_experiment(
    input_path: Path, category: str, stockout_tolerance: float = 0.02
) -> tuple[Path, Path]:
    frame = pd.read_csv(input_path)
    validation = validate_experiment_frame(frame)
    validation_path = ARTIFACT_DIR / f"{category}_experiment_intake_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    analysis_path = ARTIFACT_DIR / f"{category}_randomized_experiment_analysis.json"
    if not validation["intake_passed"]:
        analysis = {
            "status": "blocked_before_analysis",
            "category": category,
            "input": str(input_path),
            "intake_validation": validation_path.name,
            "policy_eligible_arms": [],
        }
        analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        return validation_path, analysis_path

    work = frame.copy()
    for column in (
        "assigned_treatment",
        "offered_price",
        "units_sold",
        "replacement_unit_cost",
        "supplier_funding_per_unit",
        "basket_margin",
        "category_substitution_units",
    ):
        work[column] = pd.to_numeric(work[column], errors="raise")
    work["stockout_flag"] = _boolean_series(work["stockout_flag"]).astype(int)
    work["contribution_margin"] = work["units_sold"] * (
        work["offered_price"]
        - work["replacement_unit_cost"]
        + work["supplier_funding_per_unit"]
    )
    outcomes = {
        "incremental_units": _itt_by_arm(work, "units_sold"),
        "incremental_contribution_margin": _itt_by_arm(work, "contribution_margin"),
        "incremental_basket_margin": _itt_by_arm(work, "basket_margin"),
        "incremental_category_substitution_units": _itt_by_arm(
            work, "category_substitution_units"
        ),
        "incremental_stockout_probability": _itt_by_arm(work, "stockout_flag"),
    }
    margin_by_arm = {row["arm"]: row for row in outcomes["incremental_contribution_margin"]}
    stockout_by_arm = {row["arm"]: row for row in outcomes["incremental_stockout_probability"]}
    eligible = [
        arm
        for arm, margin in margin_by_arm.items()
        if margin["ci95_low"] > 0 and stockout_by_arm[arm]["ci95_high"] <= stockout_tolerance
    ]
    analysis = {
        "status": "analyzed_randomized_intent_to_treat",
        "category": category,
        "input": str(input_path),
        "estimand": "assignment-arm intent-to-treat effect versus business-as-usual control",
        "method": "OLS with randomization-block fixed effects and store-clustered uncertainty",
        "rows": int(len(work)),
        "stores": int(work["store"].nunique()),
        "outcomes": outcomes,
        "stockout_tolerance_probability_points": stockout_tolerance,
        "policy_eligible_arms": eligible,
        "release_rule": (
            "An arm requires a positive lower 95% confidence bound for incremental contribution "
            "margin and an upper 95% confidence bound for incremental stockout probability no "
            "greater than the registered tolerance. Other release gates remain separate."
        ),
    }
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return validation_path, analysis_path


def _stores_per_arm(
    sigma: float,
    effect_log_points: float,
    weeks_per_store: int,
    intraclass_correlation: float,
    z_alpha: float = 1.96,
    z_power: float = 0.84,
) -> int:
    independent_rows = 2 * (z_alpha + z_power) ** 2 * sigma**2 / effect_log_points**2
    design_effect = 1 + (weeks_per_store - 1) * intraclass_correlation
    return math.ceil(independent_rows * design_effect / weeks_per_store)


def run_experiment_plan(category: str = "cereal") -> Path:
    calibration = json.loads(
        (ARTIFACT_DIR / f"{category}_predictive_calibration.json").read_text(encoding="utf-8")
    )
    sigma = float(calibration["log_scale_error"]["rmse"])
    scenarios = []
    for lift in (0.10, 0.20, 0.30):
        effect = float(np.log1p(lift))
        for icc in (0.01, 0.05, 0.10):
            scenarios.append(
                {
                    "minimum_detectable_unit_lift": lift,
                    "effect_log_points": effect,
                    "intraclass_correlation_assumption": icc,
                    "weeks_per_store": 13,
                    "stores_per_arm": _stores_per_arm(sigma, effect, 13, icc),
                }
            )
    plan = {
        "status": "planning_template_not_executed",
        "business_decision": (
            "Estimate which explicit offer depths create incremental contribution margin after "
            "supplier funding, substitution, stockouts, and post-promotion payback."
        ),
        "randomization_unit": "eligible store-item-week block, with store-level assignment within each wave to reduce contamination",
        "arms": [
            "business-as-usual/no targeted offer",
            "5 percent explicit price reduction",
            "10 percent explicit price reduction",
            "15 percent explicit price reduction",
        ],
        "stratification": [
            "baseline item velocity",
            "store volume tier",
            "prior promotion rate",
            "calendar wave",
        ],
        "co_primary_outcomes": [
            "incremental units including zero-sales weeks",
            "incremental contribution margin using replacement cost and supplier funding",
        ],
        "guardrails": [
            "stockout rate",
            "customer transactions and basket margin",
            "same-category substitution",
            "weeks 1-8 post-promotion pull-forward",
        ],
        "power_sensitivity": {
            "outcome_sigma_proxy": sigma,
            "alpha_two_sided": 0.05,
            "power": 0.80,
            "formula_status": "planning approximation; validate ICC and margin variance in a pilot",
            "scenarios": scenarios,
        },
        "analysis": (
            "Intent-to-treat with randomization-block fixed effects and store-clustered uncertainty; "
            "report unit and margin effects together and estimate post-period payback."
        ),
        "stop_rule": (
            "Do not scale an arm whose lower confidence bound for incremental margin is below the "
            "pre-registered downside tolerance or whose stockout/substitution guardrail fails."
        ),
    }
    path = ARTIFACT_DIR / f"{category}_randomized_experiment_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return path
