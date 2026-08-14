from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = Path(__file__).resolve().parent / "data.json"


def read_json(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        return {"available": False, "missing": name}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(name: str, limit: int | None = None) -> list[dict]:
    path = ARTIFACTS / name
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit)
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def optional_multicategory() -> dict:
    summaries = sorted(ARTIFACTS.glob("multicategory_*_summary.json"))
    if not summaries:
        return {"available": False}
    summary_path = summaries[-1]
    stem = summary_path.name.removesuffix("_summary.json")
    return {
        "available": True,
        "summary": read_json(summary_path.name),
        "categories": read_csv(f"{stem}_category_elasticities.csv"),
        "products": read_csv(f"{stem}_product_elasticities.csv"),
        "stores": read_csv(f"{stem}_store_elasticity_offsets.csv"),
        "holdout": read_json(f"{stem}_holdout_evaluation.json"),
        "holdout_by_category": read_csv(f"{stem}_holdout_by_category.csv"),
    }


def main() -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": read_json("cereal_validation.json"),
        "data_foundation": {
            "categories": [
                read_json("cereal_validation.json"),
                read_json("canned_soup_validation.json"),
                read_json("soft_drinks_validation.json"),
            ],
            "sql_mart": read_json("cereal_sql_mart_validation.json"),
        },
        "bayesian": read_json("cereal_hierarchical_summary.json"),
        "product_elasticities": read_csv("cereal_product_elasticities.csv"),
        "calibration": read_json("cereal_predictive_calibration.json"),
        "calibration_by_product": read_csv("cereal_predictive_calibration_by_product.csv"),
        "sensitivity": read_json("cereal_prior_sensitivity.json"),
        "sensitivity_by_product": read_csv("cereal_prior_sensitivity_by_product.csv"),
        "optimization": read_json("cereal_optimization_summary.json"),
        "promotion_candidates": read_csv("cereal_promotion_candidates.csv"),
        "event_summary": read_json("cereal_S_event_summary.json"),
        "event_profile": read_csv("cereal_S_isolated_event_profile.csv"),
        "payback_summary": read_json("cereal_price_promotion_end_summary.json"),
        "payback_profile": read_csv("cereal_price_promotion_end_profile.csv"),
        "causal": read_json("cereal_causal_summary.json"),
        "causal_balance": read_csv("cereal_causal_balance.csv"),
        "decision_gate": read_json("cereal_causal_decision_gate.json"),
        "causal_validation": read_json("causal_implementation_validation.json"),
        "experiment_plan": read_json("cereal_randomized_experiment_plan.json"),
        "experiment_intake": read_json("cereal_experiment_intake_validation.json"),
        "experiment_analysis": read_json("cereal_randomized_experiment_analysis.json"),
        "release_gate": read_json("cereal_release_gate.json"),
        "shadow_replay": read_json("cereal_historical_shadow_replay.json"),
        "shadow_batches": read_csv("cereal_historical_shadow_batches.csv"),
        "operational_readiness": read_json("operational_readiness.json"),
        "multicategory": optional_multicategory(),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
