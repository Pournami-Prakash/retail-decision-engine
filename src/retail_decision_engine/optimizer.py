from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from .bayesian import _prepare_cohort
from .config import ARTIFACT_DIR


@dataclass
class OptimizationResult:
    policy: str
    selected: pd.DataFrame
    objective_value: float


def _break_even_elasticity(gross_margin_rate: float, discount: float) -> float:
    """Exact constant-elasticity threshold for preserving contribution margin."""
    if discount >= gross_margin_rate:
        return float("nan")
    return float(np.log(gross_margin_rate / (gross_margin_rate - discount)) / np.log(1 - discount))


def _solve_candidates(
    candidates: pd.DataFrame,
    objective_column: str,
    policy: str,
    max_promotions: int,
    max_per_store: int,
    max_per_product: int,
    require_positive_risk_floor: bool,
    risk_floor_column: str = "incremental_margin_p10",
) -> OptimizationResult:
    eligible = candidates.copy()
    if require_positive_risk_floor:
        eligible = eligible.loc[eligible[risk_floor_column] > 0].copy()
    eligible = eligible.loc[eligible[objective_column] > 0].reset_index(drop=True)
    if eligible.empty:
        return OptimizationResult(policy, eligible, 0.0)

    model = cp_model.CpModel()
    variables = [model.new_bool_var(f"candidate_{index}") for index in eligible.index]
    scale = 1000
    coefficients = (eligible[objective_column] * scale).round().astype(int).tolist()
    model.maximize(
        sum(coefficient * variable for coefficient, variable in zip(coefficients, variables))
    )

    model.add(sum(variables) <= max_promotions)
    for _, indices in eligible.groupby(["store", "upc"]).groups.items():
        model.add(sum(variables[index] for index in indices) <= 1)
    for _, indices in eligible.groupby("store").groups.items():
        model.add(sum(variables[index] for index in indices) <= max_per_store)
    for _, indices in eligible.groupby("upc").groups.items():
        model.add(sum(variables[index] for index in indices) <= max_per_product)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Optimizer did not find a solution for {policy}: status={status}")
    selected_indices = [index for index, variable in enumerate(variables) if solver.value(variable)]
    selected = eligible.iloc[selected_indices].copy()
    return OptimizationResult(policy, selected, float(selected[objective_column].sum()))


def run_promotion_optimizer(
    category: str,
    n_products: int = 8,
    n_stores: int = 12,
    history_weeks: int = 156,
    holdout_weeks: int = 26,
    recent_weeks: int = 8,
    discounts: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    max_promotions: int = 12,
    max_per_store: int = 2,
    max_per_product: int = 4,
    seed: int = 20260723,
) -> tuple[Path, Path]:
    posterior_path = ARTIFACT_DIR / f"{category}_hierarchical_posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing {posterior_path}; run the Bayesian model first")
    idata = az.from_netcdf(posterior_path)
    train, _, products, stores, product_names = _prepare_cohort(
        category, n_products, n_stores, history_weeks, holdout_weeks
    )

    last_week = int(train["week"].max())
    recent = train.loc[train["week"] > last_week - recent_weeks].copy()
    base = recent.groupby(["upc", "store"], as_index=False).agg(
        base_units=("move", "median"),
        base_price=("unit_price", "median"),
        gross_margin_rate=("gross_margin_rate", "median"),
        observations=("week", "size"),
    )
    base = base.loc[
        (base["observations"] >= max(3, recent_weeks // 2))
        & base["gross_margin_rate"].between(0.01, 0.80)
    ].copy()
    base["description"] = base["upc"].map(product_names)
    product_index = {upc: index for index, upc in enumerate(products)}
    elasticity = idata.posterior["elasticity_product"].stack(sample=("chain", "draw"))
    rng = np.random.default_rng(seed)
    recent_groups = {
        (int(upc), int(store)): group.reset_index(drop=True)
        for (upc, store), group in recent.groupby(["upc", "store"], sort=False)
    }

    rows: list[dict[str, float | int | str]] = []
    for record in base.itertuples(index=False):
        samples = elasticity.isel(product=product_index[int(record.upc)]).to_numpy()
        history = recent_groups[(int(record.upc), int(record.store))]
        eligible_history = history.loc[history["gross_margin_rate"].between(0.01, 0.80)]
        if not eligible_history.empty:
            history = eligible_history.reset_index(drop=True)
        sampled_rows = history.iloc[rng.integers(0, len(history), size=len(samples))]
        sampled_base_units = sampled_rows["move"].to_numpy(dtype=float)
        sampled_base_price = sampled_rows["unit_price"].to_numpy(dtype=float)
        sampled_margin_rate = sampled_rows["gross_margin_rate"].to_numpy(dtype=float)
        sampled_base_cost = sampled_base_price * (1 - sampled_margin_rate)
        base_revenue = sampled_base_units * sampled_base_price
        base_margin = sampled_base_units * (sampled_base_price - sampled_base_cost)
        for discount in discounts:
            new_price = sampled_base_price * (1 - discount)
            unit_multiplier = np.power(1 - discount, samples)
            new_units = sampled_base_units * unit_multiplier
            new_revenue = new_units * new_price
            new_margin = new_units * (new_price - sampled_base_cost)
            incremental_revenue = new_revenue - base_revenue
            incremental_margin = new_margin - base_margin
            new_unit_margin = new_price - sampled_base_cost
            positive_unit_margin = new_unit_margin > 0
            required_extra_lift = np.full_like(new_margin, np.inf)
            required_extra_lift[positive_unit_margin] = (
                base_margin[positive_unit_margin] / new_margin[positive_unit_margin] - 1
            )
            break_even_elasticity = _break_even_elasticity(record.gross_margin_rate, discount)
            required_vendor_funding = np.maximum(0.0, base_margin / new_units - new_unit_margin)
            candidate = {
                "upc": int(record.upc),
                "description": record.description,
                "store": int(record.store),
                "discount": discount,
                "base_units": float(record.base_units),
                "base_price": float(record.base_price),
                "gross_margin_rate": float(record.gross_margin_rate),
                "new_unit_margin_before_funding": float(np.median(new_unit_margin)),
                "break_even_elasticity_if_margin_positive": float(break_even_elasticity),
                "expected_unit_multiplier": float(unit_multiplier.mean()),
                "incremental_revenue_mean": float(incremental_revenue.mean()),
                "incremental_margin_mean": float(incremental_margin.mean()),
                "incremental_margin_p10": float(np.quantile(incremental_margin, 0.10)),
                "probability_positive_margin": float((incremental_margin > 0).mean()),
                "required_additional_nonprice_lift_mean": float(
                    np.mean(required_extra_lift[np.isfinite(required_extra_lift)])
                )
                if np.isfinite(required_extra_lift).any()
                else float("inf"),
                "required_vendor_funding_per_unit_mean": float(required_vendor_funding.mean()),
            }
            for funding in (0.10, 0.25):
                funded_margin = incremental_margin + funding * new_units
                suffix = str(int(funding * 100)).zfill(3)
                candidate[f"incremental_margin_funding_{suffix}_mean"] = float(funded_margin.mean())
                candidate[f"incremental_margin_funding_{suffix}_p10"] = float(
                    np.quantile(funded_margin, 0.10)
                )
                candidate[f"probability_positive_margin_funding_{suffix}"] = float(
                    (funded_margin > 0).mean()
                )
            rows.append(candidate)
    candidates = pd.DataFrame(rows)
    candidate_path = ARTIFACT_DIR / f"{category}_promotion_candidates.csv"
    candidates.to_csv(candidate_path, index=False)

    margin_result = _solve_candidates(
        candidates,
        "incremental_margin_mean",
        "downside-screened contribution-margin scenario",
        max_promotions,
        max_per_store,
        max_per_product,
        require_positive_risk_floor=True,
    )
    revenue_result = _solve_candidates(
        candidates,
        "incremental_revenue_mean",
        "expected-revenue scenario",
        max_promotions,
        max_per_store,
        max_per_product,
        require_positive_risk_floor=False,
    )
    selected = pd.concat(
        [
            margin_result.selected.assign(policy=margin_result.policy),
            revenue_result.selected.assign(policy=revenue_result.policy),
        ],
        ignore_index=True,
    )
    selected_path = ARTIFACT_DIR / f"{category}_promotion_recommendations.csv"
    selected.to_csv(selected_path, index=False)

    def policy_summary(result: OptimizationResult) -> dict[str, object]:
        selected_frame = result.selected
        return {
            "selected_promotions": int(len(selected_frame)),
            "objective_value": result.objective_value,
            "incremental_revenue_mean": float(selected_frame["incremental_revenue_mean"].sum())
            if len(selected_frame)
            else 0.0,
            "incremental_margin_mean": float(selected_frame["incremental_margin_mean"].sum())
            if len(selected_frame)
            else 0.0,
            "minimum_candidate_margin_p10": float(selected_frame["incremental_margin_p10"].min())
            if len(selected_frame)
            else None,
        }

    capacity_profiles = {
        "conservative": (6, 1, 2),
        "base": (max_promotions, max_per_store, max_per_product),
        "expanded": (24, 3, 6),
    }
    capacity_sensitivity = {}
    for name, (profile_max, profile_store, profile_product) in capacity_profiles.items():
        profile_margin = _solve_candidates(
            candidates,
            "incremental_margin_mean",
            f"{name} downside-screened margin scenario",
            profile_max,
            profile_store,
            profile_product,
            require_positive_risk_floor=True,
        )
        profile_revenue = _solve_candidates(
            candidates,
            "incremental_revenue_mean",
            f"{name} expected-revenue scenario",
            profile_max,
            profile_store,
            profile_product,
            require_positive_risk_floor=False,
        )
        capacity_sensitivity[name] = {
            "constraints": {
                "max_promotions": profile_max,
                "max_per_store": profile_store,
                "max_per_product": profile_product,
            },
            "downside_screened_margin": policy_summary(profile_margin),
            "revenue": policy_summary(profile_revenue),
        }

    funding_sensitivity = {}
    for funding in (0.00, 0.10, 0.25):
        if funding == 0:
            mean_column = "incremental_margin_mean"
            p10_column = "incremental_margin_p10"
        else:
            suffix = str(int(funding * 100)).zfill(3)
            mean_column = f"incremental_margin_funding_{suffix}_mean"
            p10_column = f"incremental_margin_funding_{suffix}_p10"
        funded_result = _solve_candidates(
            candidates,
            mean_column,
            f"vendor-funding-{funding:.2f} downside-screened margin scenario",
            max_promotions,
            max_per_store,
            max_per_product,
            require_positive_risk_floor=True,
            risk_floor_column=p10_column,
        )
        funding_sensitivity[f"{funding:.2f}"] = {
            "funding_per_promoted_unit": funding,
            "selected_promotions": int(len(funded_result.selected)),
            "funded_incremental_margin_mean": funded_result.objective_value,
            "minimum_funded_margin_p10": (
                float(funded_result.selected[p10_column].min())
                if len(funded_result.selected)
                else None
            ),
        }

    summary = {
        "category": category,
        "decision_question": (
            "Which store-product discounts maximize revenue versus contribution margin under "
            "portfolio-capacity and posterior-risk constraints?"
        ),
        "guardrails": [
            "Uses price-response posterior only; observed promotion-code effects are excluded because they are not causal.",
            "Zero selected actions means price-cut-only scenarios fail the stated screen; it does not imply that fully measured display, feature, loyalty, basket, or funded promotions are unprofitable.",
            "Excluded promotion-code associations can mix genuine nonprice lift with endogenous targeting, so the direction of omitted causal merchandising lift is unknown.",
            "Unit cost is approximated from historical gross-margin rate and base price.",
            "The data manual warns accounting acquisition cost is not replacement cost.",
            "No-action is an allowed and economically meaningful optimizer result.",
            "Portfolio limits are illustrative scenario inputs, not stakeholder-approved operating constraints.",
            "Bootstrap draws propagate recent baseline volume, price, and accounting-margin variation alongside elasticity uncertainty.",
            "Residual demand shocks, substitution, stockouts, supplier funding, and future covariate uncertainty remain outside the scenario distribution.",
        ],
        "candidate_count": int(len(candidates)),
        "constraints": {
            "max_promotions": max_promotions,
            "max_per_store": max_per_store,
            "max_per_product": max_per_product,
            "risk_aware_margin_policy_requires_p10_above_zero": True,
        },
        "policies": {
            "risk_aware_margin": policy_summary(margin_result),
            "revenue": policy_summary(revenue_result),
        },
        "capacity_sensitivity": capacity_sensitivity,
        "vendor_funding_sensitivity": funding_sensitivity,
        "decision_use": (
            "Assumption-transparent price-cut-only scenario comparison for experiment planning; "
            "not a verdict on fully bundled promotions and not a validated or deployable pricing policy."
        ),
    }
    summary_path = ARTIFACT_DIR / f"{category}_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path, selected_path
