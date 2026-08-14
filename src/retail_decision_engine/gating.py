from __future__ import annotations

import json
from pathlib import Path

from .config import ARTIFACT_DIR


def run_causal_decision_gate(category: str) -> Path:
    causal_path = ARTIFACT_DIR / f"{category}_causal_summary.json"
    optimization_path = ARTIFACT_DIR / f"{category}_optimization_summary.json"
    with causal_path.open(encoding="utf-8") as handle:
        causal = json.load(handle)
    with optimization_path.open(encoding="utf-8") as handle:
        observational_optimization = json.load(handle)

    passed = bool(causal["identification_gate_passed"])
    if passed:
        status = "causal_estimate_eligible_for_policy_research"
        explanation = (
            "Identification diagnostics passed. A new optimizer must still be fit to treatment-specific "
            "price depth, supplier funding, and unit-cost data before operational use."
        )
    else:
        status = "causal_estimate_withheld_from_optimizer"
        explanation = (
            "The AIPW point estimate is not admitted because overlap, balance, placebo, or negative-control "
            "requirements failed. Existing optimization remains an explicitly associational price-response scenario."
        )

    diagnostics = causal["diagnostics"]
    result = {
        "category": category,
        "status": status,
        "gate_passed": passed,
        "explanation": explanation,
        "failed_or_risky_checks": {
            "overlap_fraction_below_080": diagnostics["fraction_strictly_inside_005_095"] < 0.80,
            "post_weighting_smd_above_010": diagnostics["max_abs_smd_after_ipw"] >= 0.10,
            "future_treatment_placebo_excludes_zero": not (
                causal["future_treatment_lead4_placebo"]["ci95_low_log_points"]
                <= 0
                <= causal["future_treatment_lead4_placebo"]["ci95_high_log_points"]
            ),
            "past_outcome_negative_control_excludes_zero": not (
                causal["past_outcome_negative_control"]["ci95_low_log_points"]
                <= 0
                <= causal["past_outcome_negative_control"]["ci95_high_log_points"]
            ),
        },
        "observational_optimizer_status": {
            "risk_aware_margin": observational_optimization["policies"]["risk_aware_margin"],
            "revenue": observational_optimization["policies"]["revenue"],
        },
        "data_required_for_causal_optimization": [
            "Known randomized or plausibly exogenous promotion assignment",
            "Explicit discount depth and display/feature treatment fields",
            "Supplier funding and true replacement-cost data",
            "Inventory and stockout indicators",
            "Zero-sales observations with valid offered prices",
        ],
    }
    output = ARTIFACT_DIR / f"{category}_causal_decision_gate.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
