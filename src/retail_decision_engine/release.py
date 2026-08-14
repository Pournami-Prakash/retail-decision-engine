from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ARTIFACT_DIR


def _read(name: str) -> dict[str, Any] | None:
    path = ARTIFACT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_release_gate(category: str, coverage_target: float = 0.90) -> Path:
    intake = _read(f"{category}_experiment_intake_validation.json")
    experiment = _read(f"{category}_randomized_experiment_analysis.json")
    calibration = _read(f"{category}_predictive_calibration.json")
    historical_causal = _read(f"{category}_causal_decision_gate.json")
    monitoring = _read(f"{category}_live_monitoring.json")

    conformal_coverage = None
    if calibration:
        conformal_coverage = calibration["temporal_split_conformal_recalibration"]["intervals"][
            "90"
        ]["evaluation_coverage"]
    eligible_arms = experiment.get("policy_eligible_arms", []) if experiment else []
    checks = {
        "experiment_intake_complete": bool(intake and intake.get("intake_passed")),
        "randomized_itt_analysis_complete": bool(
            experiment and experiment.get("status") == "analyzed_randomized_intent_to_treat"
        ),
        "positive_margin_arm_within_stockout_guardrail": bool(eligible_arms),
        "predictive_coverage_meets_target": bool(
            conformal_coverage is not None and conformal_coverage >= coverage_target
        ),
        "live_monitoring_passes": bool(
            monitoring and monitoring.get("overall_status") == "pass"
        ),
    }
    cleared = all(checks.values())
    blockers = [name for name, passed in checks.items() if not passed]
    result = {
        "category": category,
        "release_decision": "cleared_for_controlled_rollout" if cleared else "blocked",
        "cleared": cleared,
        "checks": checks,
        "blockers": blockers,
        "evidence": {
            "experiment_intake_status": intake.get("status") if intake else "not_received",
            "experiment_analysis_status": experiment.get("status") if experiment else "not_run",
            "policy_eligible_arms": eligible_arms,
            "conformal_90_evaluation_coverage": conformal_coverage,
            "coverage_target": coverage_target,
            "live_monitoring_status": monitoring.get("overall_status")
            if monitoring
            else "not_evaluated",
            "historical_observational_gate": historical_causal.get("status")
            if historical_causal
            else "not_run",
        },
        "decision_logic": (
            "Historical observational estimates never clear this gate. Release requires complete "
            "randomized evidence, an arm with downside-safe incremental margin and stockouts, "
            "predictive calibration at target, and passing live-batch monitoring."
        ),
    }
    path = ARTIFACT_DIR / f"{category}_release_gate.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path
