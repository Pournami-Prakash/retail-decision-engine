from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import ARTIFACT_DIR, PROCESSED_DIR


FEATURES = [
    "lag1_log_units",
    "lag2_log_units",
    "rolling4_log_units",
    "lag1_log_price",
    "rolling4_log_price",
    "prior_panel_log_units",
    "prior_promotion_rate",
    "season_sin",
    "season_cos",
    "trend",
]
NEGATIVE_CONTROL_FEATURES = [
    "nc_lag3_log_units",
    "nc_rolling4_log_units",
    "nc_lag3_log_price",
    "nc_rolling4_log_price",
    "nc_prior_panel_log_units",
    "nc_prior_promotion_rate",
    "nc_outcome_season_sin",
    "nc_outcome_season_cos",
    "nc_outcome_trend",
]


def _prior_expanding_mean(values: pd.Series) -> pd.Series:
    count = np.arange(len(values), dtype=float)
    cumulative = values.cumsum().shift(1)
    return cumulative / count


def _prepare_causal_cohort(
    category: str,
    history_weeks: int = 180,
    n_products: int = 30,
) -> pd.DataFrame:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    frame = pd.read_csv(
        panel_path,
        usecols=["upc", "store", "week", "move", "unit_price", "sale", "descrip"],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )
    frame = frame.loc[(frame["move"] > 0) & (frame["unit_price"] > 0)].copy()
    max_week = int(frame["week"].max())
    frame = frame.loc[frame["week"] > max_week - history_weeks].copy()
    products = frame.groupby("upc")["move"].sum().nlargest(n_products).index
    frame = frame.loc[frame["upc"].isin(products)].copy()
    frame["sale_code"] = frame["sale"].fillna("").astype(str).str.strip().str.upper()
    frame = frame.loc[frame["sale_code"].isin(["", "S"])].copy()
    frame["treatment"] = frame["sale_code"].eq("S").astype(int)
    frame["log_units"] = np.log(frame["move"])
    frame["log_price"] = np.log(frame["unit_price"])
    frame["panel_id"] = frame["store"].astype(str) + ":" + frame["upc"].astype(str)
    frame = frame.sort_values(["panel_id", "week"]).reset_index(drop=True)

    group = frame.groupby("panel_id", sort=False)
    frame["lag1_week"] = group["week"].shift(1)
    frame["lag2_week"] = group["week"].shift(2)
    frame["lag6_week"] = group["week"].shift(6)
    frame["lead4_week"] = group["week"].shift(-4)
    frame["lag1_log_units"] = group["log_units"].shift(1)
    frame["lag2_log_units"] = group["log_units"].shift(2)
    frame["lag1_log_price"] = group["log_price"].shift(1)
    frame["rolling4_log_units"] = group["log_units"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=3).mean()
    )
    frame["rolling4_log_price"] = group["log_price"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=3).mean()
    )
    frame["prior_panel_log_units"] = group["log_units"].transform(_prior_expanding_mean)
    frame["prior_promotion_rate"] = group["treatment"].transform(_prior_expanding_mean)
    frame["lead_treatment_4"] = group["treatment"].shift(-4)
    frame.loc[frame["lead4_week"] - frame["week"] != 4, "lead_treatment_4"] = np.nan

    continuous = (frame["week"] - frame["lag1_week"] == 1) & (
        frame["week"] - frame["lag2_week"] == 2
    )
    frame["season_sin"] = np.sin(2 * np.pi * frame["week"] / 52.0)
    frame["season_cos"] = np.cos(2 * np.pi * frame["week"] / 52.0)
    frame["trend"] = (frame["week"] - frame["week"].min()) / history_weeks
    frame["nc_lag3_log_units"] = group["log_units"].shift(3)
    frame["nc_rolling4_log_units"] = group["log_units"].transform(
        lambda values: values.shift(3).rolling(4, min_periods=4).mean()
    )
    frame["nc_lag3_log_price"] = group["log_price"].shift(3)
    frame["nc_rolling4_log_price"] = group["log_price"].transform(
        lambda values: values.shift(3).rolling(4, min_periods=4).mean()
    )
    frame["nc_prior_panel_log_units"] = group["log_units"].transform(
        lambda values: values.expanding().mean().shift(3)
    )
    frame["nc_prior_promotion_rate"] = group["treatment"].transform(
        lambda values: values.expanding().mean().shift(3)
    )
    frame["nc_outcome_season_sin"] = np.sin(2 * np.pi * (frame["week"] - 2) / 52.0)
    frame["nc_outcome_season_cos"] = np.cos(2 * np.pi * (frame["week"] - 2) / 52.0)
    frame["nc_outcome_trend"] = (frame["week"] - 2 - frame["week"].min()) / history_weeks
    negative_history_continuous = frame["week"] - frame["lag6_week"] == 6
    frame.loc[~negative_history_continuous, NEGATIVE_CONTROL_FEATURES] = np.nan
    frame = frame.loc[continuous].dropna(subset=FEATURES).reset_index(drop=True)
    return frame


def _clustered_mean_interval(score: np.ndarray, clusters: pd.Series) -> dict[str, float]:
    estimate = float(score.mean())
    centered = score - estimate
    cluster_sums = pd.Series(centered).groupby(clusters.reset_index(drop=True)).sum().to_numpy()
    n = len(score)
    groups = len(cluster_sums)
    variance = float(np.sum(cluster_sums**2) / (n**2))
    if groups > 1:
        variance *= groups / (groups - 1)
    standard_error = float(np.sqrt(variance))
    return {
        "estimate_log_points": estimate,
        "estimated_unit_difference_pct": float(100 * np.expm1(estimate)),
        "clustered_standard_error": standard_error,
        "ci95_low_log_points": estimate - 1.96 * standard_error,
        "ci95_high_log_points": estimate + 1.96 * standard_error,
        "clusters": groups,
        "rows": n,
    }


def _weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def _balance_table(
    frame: pd.DataFrame, treatment: np.ndarray, propensity: np.ndarray
) -> pd.DataFrame:
    rows = []
    treated = treatment == 1
    control = ~treated
    weights_t = 1 / propensity[treated]
    weights_c = 1 / (1 - propensity[control])
    for feature in FEATURES:
        values = frame[feature].to_numpy()
        mean_t, var_t = _weighted_mean_variance(values[treated], np.ones(treated.sum()))
        mean_c, var_c = _weighted_mean_variance(values[control], np.ones(control.sum()))
        pooled = np.sqrt((var_t + var_c) / 2)
        before = (mean_t - mean_c) / pooled if pooled > 0 else 0.0
        wmean_t, wvar_t = _weighted_mean_variance(values[treated], weights_t)
        wmean_c, wvar_c = _weighted_mean_variance(values[control], weights_c)
        wpooled = np.sqrt((wvar_t + wvar_c) / 2)
        after = (wmean_t - wmean_c) / wpooled if wpooled > 0 else 0.0
        rows.append(
            {
                "feature": feature,
                "smd_before": before,
                "smd_after_ipw": after,
                "abs_smd_before": abs(before),
                "abs_smd_after_ipw": abs(after),
            }
        )
    return pd.DataFrame(rows)


def _crossfit_aipw(
    frame: pd.DataFrame,
    treatment_column: str,
    outcome_column: str,
    features: list[str] | None = None,
    folds: int = 4,
    split_strategy: str = "panel",
    seed: int = 20260720,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = FEATURES if features is None else features
    usable = frame.dropna(subset=[treatment_column, outcome_column, *features]).copy()
    treatment = usable[treatment_column].astype(int).to_numpy()
    outcome = usable[outcome_column].to_numpy()
    x = usable[features].to_numpy()
    if split_strategy == "panel":
        panels = np.sort(usable["panel_id"].unique())
        shuffled = np.random.default_rng(seed).permutation(panels)
        panel_folds = {panel: index % folds for index, panel in enumerate(shuffled)}
        fold_id = usable["panel_id"].map(panel_folds).to_numpy()
    elif split_strategy == "contiguous_week":
        week_blocks = np.array_split(np.sort(usable["week"].unique()), folds)
        week_folds = {int(week): fold for fold, block in enumerate(week_blocks) for week in block}
        fold_id = usable["week"].map(week_folds).to_numpy()
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    propensity = np.zeros(len(usable))
    outcome_one = np.zeros(len(usable))
    outcome_zero = np.zeros(len(usable))
    for fold in range(folds):
        test = fold_id == fold
        train = ~test
        propensity_model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=24,
            min_samples_leaf=100,
            l2_regularization=1.0,
            random_state=seed + fold,
        )
        propensity_model.fit(x[train], treatment[train])
        propensity[test] = propensity_model.predict_proba(x[test])[:, 1]

        for arm, destination in ((1, outcome_one), (0, outcome_zero)):
            arm_train = train & (treatment == arm)
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=24,
                min_samples_leaf=80 if arm else 150,
                l2_regularization=1.0,
                random_state=seed + 10 + fold + arm,
            )
            model.fit(x[arm_train], outcome[arm_train])
            destination[test] = model.predict(x[test])

    raw_propensity = propensity.copy()
    propensity = np.clip(raw_propensity, 0.05, 0.95)
    score = (
        outcome_one
        - outcome_zero
        + treatment * (outcome - outcome_one) / propensity
        - (1 - treatment) * (outcome - outcome_zero) / (1 - propensity)
    )
    return score, propensity, raw_propensity, treatment, usable.index.to_numpy()


def run_doubly_robust_promotion_analysis(
    category: str,
    history_weeks: int = 180,
    n_products: int = 30,
) -> tuple[Path, Path, Path]:
    frame = _prepare_causal_cohort(category, history_weeks, n_products)
    primary_score, propensity, raw_propensity, treatment, primary_index = _crossfit_aipw(
        frame, "treatment", "log_units", split_strategy="panel"
    )
    primary_frame = frame.loc[primary_index].reset_index(drop=True)
    primary = _clustered_mean_interval(primary_score, primary_frame["panel_id"])

    overlap = (raw_propensity > 0.05) & (raw_propensity < 0.95)
    primary_overlap = _clustered_mean_interval(
        primary_score[overlap], primary_frame.loc[overlap, "panel_id"]
    )
    balance = _balance_table(primary_frame, treatment, propensity)
    balance_path = ARTIFACT_DIR / f"{category}_causal_balance.csv"
    balance.to_csv(balance_path, index=False)

    time_score, _, _, _, time_index = _crossfit_aipw(
        frame,
        "treatment",
        "log_units",
        split_strategy="contiguous_week",
        seed=20260725,
    )
    time_frame = frame.loc[time_index].reset_index(drop=True)
    time_sensitivity = _clustered_mean_interval(time_score, time_frame["panel_id"])

    lead_score, _, _, _, lead_index = _crossfit_aipw(
        frame, "lead_treatment_4", "log_units", seed=20260730
    )
    lead_frame = frame.loc[lead_index].reset_index(drop=True)
    lead_placebo = _clustered_mean_interval(lead_score, lead_frame["panel_id"])

    negative_score, _, _, _, negative_index = _crossfit_aipw(
        frame,
        "treatment",
        "lag2_log_units",
        features=NEGATIVE_CONTROL_FEATURES,
        seed=20260740,
    )
    negative_frame = frame.loc[negative_index].reset_index(drop=True)
    negative_control = _clustered_mean_interval(negative_score, negative_frame["panel_id"])

    def includes_zero(result: dict[str, float]) -> bool:
        return result["ci95_low_log_points"] <= 0 <= result["ci95_high_log_points"]

    treated_propensity = raw_propensity[treatment == 1]
    control_propensity = raw_propensity[treatment == 0]
    diagnostics = {
        "primary_crossfit_unit": "store-product panel",
        "sensitivity_crossfit_unit": "contiguous calendar-week block",
        "score_propensity_clip": [0.05, 0.95],
        "treatment_rate": float(treatment.mean()),
        "propensity_auc": float(roc_auc_score(treatment, propensity)),
        "propensity_brier": float(brier_score_loss(treatment, propensity)),
        "treated_propensity_quantiles": {
            str(q): float(np.quantile(treated_propensity, q))
            for q in (0.01, 0.10, 0.50, 0.90, 0.99)
        },
        "control_propensity_quantiles": {
            str(q): float(np.quantile(control_propensity, q))
            for q in (0.01, 0.10, 0.50, 0.90, 0.99)
        },
        "fraction_strictly_inside_005_095": float(overlap.mean()),
        "max_abs_smd_before": float(balance["abs_smd_before"].max()),
        "max_abs_smd_after_ipw": float(balance["abs_smd_after_ipw"].max()),
    }
    passed = (
        diagnostics["fraction_strictly_inside_005_095"] >= 0.80
        and diagnostics["max_abs_smd_after_ipw"] < 0.10
        and includes_zero(lead_placebo)
        and includes_zero(negative_control)
    )
    summary = {
        "category": category,
        "estimand": (
            "ATE contrast of recorded simple-sale weeks versus unlabeled weeks on log "
            "positive-sale units; unlabeled weeks are not verified non-promotion controls"
        ),
        "method": (
            "four-fold store-product-panel-cross-fitted augmented inverse propensity weighting, "
            "with contiguous-week-block sensitivity"
        ),
        "cohort": {
            "history_weeks": history_weeks,
            "top_products": n_products,
            "rows": int(len(primary_frame)),
            "panels": int(primary_frame["panel_id"].nunique()),
        },
        "primary_aipw": primary,
        "contiguous_week_block_sensitivity": time_sensitivity,
        "overlap_trimmed_aipw": primary_overlap,
        "future_treatment_lead4_placebo": lead_placebo,
        "past_outcome_negative_control": negative_control,
        "past_outcome_negative_control_design": {
            "outcome_time": "t-2 log units",
            "latest_adjustment_time": "t-3",
            "history_requirement": "continuous observed weeks t-6 through t",
            "features": NEGATIVE_CONTROL_FEATURES,
            "purpose": (
                "Tests whether current recorded treatment remains associated with a temporally "
                "prior outcome after adjustment using only information preceding that outcome."
            ),
        },
        "diagnostics": diagnostics,
        "identification_gate_passed": passed,
        "decision_rule": (
            "Causal estimates are not admitted to the optimizer unless overlap/balance pass and both "
            "falsification confidence intervals include zero. Thresholds are predeclared screening "
            "rules, not universal statistical laws."
        ),
        "limitations": [
            "AIPW is doubly robust to one nuisance-model misspecification, not unmeasured confounding.",
            "The source has incomplete promotion flags, so unlabeled weeks can contain promotions.",
            "The source has only positive-price positive-sales observations.",
            "The treatment combines discount and merchandising; it does not isolate a pure price intervention.",
            "Panel and contiguous-week cross-fitting are reported as sensitivity designs; neither creates exogenous treatment assignment.",
        ],
    }
    summary_path = ARTIFACT_DIR / f"{category}_causal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    scored = primary_frame[
        ["upc", "descrip", "store", "week", "treatment", "log_units", "panel_id"]
    ].copy()
    scored["propensity"] = propensity
    scored["aipw_score"] = primary_score
    scored_path = ARTIFACT_DIR / f"{category}_causal_scored.csv.gz"
    scored.to_csv(scored_path, index=False, compression="gzip")
    return summary_path, balance_path, scored_path
