from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal import FEATURES, _crossfit_aipw
from .config import ARTIFACT_DIR


def _synthetic_panel(
    scenario: str,
    seed: int,
    panels: int = 160,
    weeks: int = 36,
    true_effect: float = 0.25,
) -> pd.DataFrame:
    """Create a known-truth panel for estimator implementation checks."""
    rng = np.random.default_rng(seed)
    panel = np.repeat(np.arange(panels), weeks)
    week = np.tile(np.arange(weeks), panels)
    n = len(panel)
    panel_effect = rng.normal(0, 0.35, panels)[panel]
    hidden = rng.normal(size=n)
    x1 = rng.normal(size=n) + 0.35 * panel_effect
    x2 = 0.45 * x1 + rng.normal(scale=0.9, size=n)
    season_sin = np.sin(2 * np.pi * week / 12)
    season_cos = np.cos(2 * np.pi * week / 12)
    feature_values = {
        "lag1_log_units": x1,
        "lag2_log_units": x2,
        "rolling4_log_units": 0.6 * x1 + rng.normal(scale=0.5, size=n),
        "lag1_log_price": rng.normal(size=n),
        "rolling4_log_price": rng.normal(size=n),
        "prior_panel_log_units": panel_effect + rng.normal(scale=0.3, size=n),
        "prior_promotion_rate": rng.uniform(0, 0.25, size=n),
        "season_sin": season_sin,
        "season_cos": season_cos,
        "trend": week / max(weeks - 1, 1),
        "nc_lag3_log_units": x1,
        "nc_rolling4_log_units": 0.6 * x1 + rng.normal(scale=0.5, size=n),
        "nc_lag3_log_price": rng.normal(size=n),
        "nc_rolling4_log_price": rng.normal(size=n),
        "nc_prior_panel_log_units": panel_effect + rng.normal(scale=0.3, size=n),
        "nc_prior_promotion_rate": rng.uniform(0, 0.25, size=n),
        "nc_outcome_season_sin": season_sin,
        "nc_outcome_season_cos": season_cos,
        "nc_outcome_trend": week / max(weeks - 1, 1),
    }
    linear_treatment = np.full(n, -1.0)
    if scenario != "randomized":
        linear_treatment += 0.75 * x1 - 0.45 * x2 + 0.35 * season_sin
    if scenario == "hidden_confounding":
        linear_treatment += 1.0 * hidden
    probability = 1 / (1 + np.exp(-linear_treatment))
    treatment = rng.binomial(1, probability)
    baseline = 1.5 + panel_effect + 0.55 * x1 - 0.25 * x2 + 0.20 * x1**2 + 0.15 * season_cos
    if scenario == "hidden_confounding":
        baseline += 0.75 * hidden
    outcome = baseline + true_effect * treatment + rng.normal(scale=0.55, size=n)
    frame = pd.DataFrame(feature_values)
    frame["panel_id"] = "p" + pd.Series(panel).astype(str)
    frame["week"] = week
    frame["treatment"] = treatment
    frame["outcome"] = outcome
    return frame


def run_causal_implementation_validation(
    seed: int = 20260720,
    true_effect: float = 0.25,
) -> Path:
    rows = []
    for offset, scenario in enumerate(("randomized", "measured_confounding", "hidden_confounding")):
        frame = _synthetic_panel(scenario, seed + offset, true_effect=true_effect)
        score, _, raw_propensity, treatment, _ = _crossfit_aipw(
            frame,
            "treatment",
            "outcome",
            features=FEATURES,
            split_strategy="panel",
            seed=seed + 100 + offset,
        )
        estimate = float(score.mean())
        rows.append(
            {
                "scenario": scenario,
                "known_true_effect_log_points": true_effect,
                "estimated_effect_log_points": estimate,
                "absolute_error": abs(estimate - true_effect),
                "treatment_rate": float(treatment.mean()),
                "strict_overlap_fraction": float(
                    ((raw_propensity > 0.05) & (raw_propensity < 0.95)).mean()
                ),
            }
        )
    by_scenario = {row["scenario"]: row for row in rows}
    observed_cases_pass = all(
        by_scenario[name]["absolute_error"] <= 0.08
        for name in ("randomized", "measured_confounding")
    )
    hidden_bias_visible = by_scenario["hidden_confounding"]["absolute_error"] >= 0.10
    output = {
        "purpose": (
            "Known-truth implementation validation. This checks estimator plumbing and shows its "
            "failure under omitted confounding; it does not validate the historical promotion effect."
        ),
        "features_exclude_true_hidden_confounder": True,
        "scenarios": rows,
        "acceptance_rules": {
            "randomized_and_measured_absolute_error_at_most": 0.08,
            "hidden_confounding_absolute_error_at_least": 0.10,
        },
        "implementation_validation_passed": observed_cases_pass and hidden_bias_visible,
    }
    path = ARTIFACT_DIR / "causal_implementation_validation.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return path
