from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ARTIFACT_DIR


ROOT = Path(__file__).resolve().parents[2]
CRITICAL_ARTIFACTS = (
    "cereal_validation.json",
    "cereal_sql_mart_validation.json",
    "cereal_hierarchical_summary.json",
    "cereal_predictive_calibration.json",
    "cereal_causal_summary.json",
    "cereal_causal_decision_gate.json",
    "cereal_optimization_summary.json",
    "cereal_randomized_experiment_plan.json",
    "cereal_experiment_intake_validation.json",
    "cereal_randomized_experiment_analysis.json",
    "cereal_release_gate.json",
    "cereal_historical_shadow_replay.json",
    "causal_implementation_validation.json",
    "multicategory_cereal_canned_soup_soft_drinks_summary.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_registry() -> Path:
    entries = []
    for name in CRITICAL_ARTIFACTS:
        path = ARTIFACT_DIR / name
        if not path.exists():
            continue
        entries.append(
            {
                "artifact": name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(entries),
        "artifacts": entries,
        "lineage_note": (
            "Hashes make the dashboard/model-card evidence traceable to immutable local outputs. "
            "A deployed registry would additionally record code commit and signed storage URI."
        ),
    }
    path = ARTIFACT_DIR / "model_registry.json"
    path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return path


def score_decision_request(request: dict[str, Any]) -> dict[str, Any]:
    required = {
        "category",
        "store",
        "upc",
        "discount",
        "replacement_unit_cost",
        "supplier_funding_per_unit",
        "inventory_available",
    }
    missing = sorted(required - request.keys())
    if missing:
        return {"status": "invalid", "reasons": [f"missing:{name}" for name in missing]}
    if not 0 < float(request["discount"]) < 1:
        return {"status": "invalid", "reasons": ["discount_must_be_between_zero_and_one"]}
    gate_path = ARTIFACT_DIR / f"{request['category']}_causal_decision_gate.json"
    if not gate_path.exists():
        return {"status": "blocked", "reasons": ["causal_gate_artifact_missing"]}
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_passed", False):
        return {
            "status": "blocked",
            "reasons": ["causal_estimate_not_cleared"],
            "gate_status": gate.get("status"),
            "decision": None,
        }
    candidates_path = ARTIFACT_DIR / f"{request['category']}_promotion_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    match = candidates.loc[
        candidates["store"].eq(int(request["store"]))
        & candidates["upc"].astype(str).eq(str(request["upc"]))
        & candidates["discount"].sub(float(request["discount"])).abs().lt(1e-9)
    ]
    if match.empty:
        return {"status": "blocked", "reasons": ["candidate_outside_validated_scope"]}
    return {
        "status": "review_required",
        "reasons": ["human_merchandising_and_finance_approval_required"],
        "decision": match.iloc[0].to_dict(),
    }


def run_operational_readiness() -> tuple[Path, Path]:
    registry_path = build_model_registry()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    fact_contract = json.loads(
        (ROOT / "contracts/store_item_week.schema.json").read_text(encoding="utf-8")
    )
    panel_columns = set(
        pd.read_csv(ROOT / "data/processed/cereal_store_product_week.csv.gz", nrows=1).columns
    )
    checks = {
        "artifact_lineage_complete": registry["artifact_count"] == len(CRITICAL_ARTIFACTS),
        "request_contract_present": (ROOT / "contracts/decision_request.schema.json").exists(),
        "response_contract_present": (ROOT / "contracts/decision_response.schema.json").exists(),
        "analytical_fact_contract_present": (
            ROOT / "contracts/store_item_week.schema.json"
        ).exists(),
        "experiment_observation_contract_present": (
            ROOT / "contracts/promotion_experiment_observation.schema.json"
        ).exists(),
        "experiment_collection_template_present": (
            ROOT / "templates/promotion_experiment_observations.csv"
        ).exists(),
        "analytical_fact_contract_matches_panel": set(fact_contract["required"]).issubset(
            panel_columns
        ),
        "continuous_integration_present": (ROOT / ".github/workflows/ci.yml").exists(),
        "container_definition_present": (ROOT / "Dockerfile").exists(),
        "container_runtime_validation_present": (ROOT / "docs/docker_validation.md").exists(),
        "fail_closed_decision_service_present": (
            ROOT / "src/retail_decision_engine/service.py"
        ).exists(),
        "monitoring_implementation_present": (
            ROOT / "src/retail_decision_engine/monitoring.py"
        ).exists(),
        "historical_shadow_replay_present": (
            ARTIFACT_DIR / "cereal_historical_shadow_replay.json"
        ).exists(),
        "model_card_present": (ROOT / "docs/model_card.md").exists(),
        "runbook_present": (ROOT / "docs/production_runbook.md").exists(),
    }
    gate = json.loads(
        (ARTIFACT_DIR / "cereal_causal_decision_gate.json").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (ARTIFACT_DIR / "cereal_predictive_calibration.json").read_text(encoding="utf-8")
    )
    release_gate = json.loads(
        (ARTIFACT_DIR / "cereal_release_gate.json").read_text(encoding="utf-8")
    )
    shadow_replay = json.loads(
        (ARTIFACT_DIR / "cereal_historical_shadow_replay.json").read_text(encoding="utf-8")
    )
    dry_run = score_decision_request(
        {
            "category": "cereal",
            "store": 8,
            "upc": 1600066590,
            "discount": 0.05,
            "replacement_unit_cost": 2.50,
            "supplier_funding_per_unit": 0.00,
            "inventory_available": True,
        }
    )
    report = {
        "system_engineering_checks": checks,
        "system_engineering_passed": all(checks.values()),
        "fail_closed_dry_run": dry_run,
        "historical_policy_clearance": {
            "cleared": bool(release_gate.get("cleared", False)),
            "causal_gate": gate.get("status"),
            "nominal_90_coverage": calibration["predictive_interval_calibration"]["90"][
                "empirical"
            ],
            "live_drift_status": "not_evaluated_no_live_batch",
            "replacement_cost_status": "not_available_in_source",
        },
        "release_gates": release_gate,
        "historical_shadow_replay": shadow_replay,
        "release_decision": release_gate["release_decision"],
        "explanation": (
            "Engineering controls can pass while the policy remains blocked. Production readiness "
            "never overrides failed causal, calibration, live-drift, or economic-data gates."
        ),
    }
    report_path = ARTIFACT_DIR / "operational_readiness.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return registry_path, report_path
