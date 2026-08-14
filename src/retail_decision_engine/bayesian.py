from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .config import ARTIFACT_DIR, PROCESSED_DIR


PROMOTION_CODES = ("B", "S", "C", "G")
PRIOR_PROFILES = {
    "regularized": {
        "mean_elasticity_mu": -1.5,
        "mean_elasticity_sigma": 1.0,
        "sigma_elasticity": 0.75,
        "promotion_sigma": 1.0,
    },
    "skeptical": {
        "mean_elasticity_mu": -1.0,
        "mean_elasticity_sigma": 0.5,
        "sigma_elasticity": 0.4,
        "promotion_sigma": 0.5,
    },
    "weak": {
        "mean_elasticity_mu": -1.5,
        "mean_elasticity_sigma": 2.5,
        "sigma_elasticity": 1.5,
        "promotion_sigma": 2.0,
    },
}


def _posterior_samples(idata: az.InferenceData, name: str) -> np.ndarray:
    values = idata.posterior[name].stack(sample=("chain", "draw"))
    if values.ndim == 1:
        return values.to_numpy()
    return values.transpose(..., "sample").to_numpy()


def _linear_predictor_draws(
    idata: az.InferenceData,
    frame: pd.DataFrame,
) -> np.ndarray:
    alpha = _posterior_samples(idata, "alpha")
    product_intercept = _posterior_samples(idata, "product_intercept")
    store_intercept = _posterior_samples(idata, "store_intercept")
    elasticity = _posterior_samples(idata, "elasticity_product")
    promo_effect = _posterior_samples(idata, "promo_effect")
    seasonal_sin = _posterior_samples(idata, "seasonal_sin")
    seasonal_cos = _posterior_samples(idata, "seasonal_cos")
    trend_effect = _posterior_samples(idata, "trend_effect")

    prediction = (
        alpha[None, :]
        + product_intercept[frame["product_idx"].to_numpy(), :]
        + store_intercept[frame["store_idx"].to_numpy(), :]
        + elasticity[frame["product_idx"].to_numpy(), :]
        * frame["centered_log_price"].to_numpy()[:, None]
        + seasonal_sin[None, :] * frame["season_sin"].to_numpy()[:, None]
        + seasonal_cos[None, :] * frame["season_cos"].to_numpy()[:, None]
        + trend_effect[None, :] * frame["trend"].to_numpy()[:, None]
    )
    promo_idx = frame["promo_idx"].to_numpy()
    promoted = promo_idx >= 0
    prediction[promoted] += promo_effect[promo_idx[promoted], :]
    return prediction


def _predict_conditional_mean_log_units(
    idata: az.InferenceData,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = _linear_predictor_draws(idata, frame)
    return (
        prediction.mean(axis=1),
        np.quantile(prediction, 0.05, axis=1),
        np.quantile(prediction, 0.95, axis=1),
    )


def _prepare_cohort(
    category: str,
    n_products: int,
    n_stores: int,
    history_weeks: int,
    holdout_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int], dict[int, str]]:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    frame = pd.read_csv(
        panel_path,
        usecols=[
            "upc",
            "store",
            "week",
            "move",
            "unit_price",
            "gross_margin_rate",
            "gross_margin_dollars",
            "sale",
            "descrip",
        ],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )
    frame = frame.loc[(frame["move"] > 0) & (frame["unit_price"] > 0)].copy()
    max_week = int(frame["week"].max())
    cutoff = max_week - holdout_weeks
    history_start = cutoff - history_weeks + 1
    research_window = frame.loc[frame["week"] >= history_start].copy()
    train_window = research_window.loc[research_window["week"] <= cutoff]

    products = (
        train_window.groupby("upc")["move"].sum().nlargest(n_products).index.astype(int).tolist()
    )
    product_names = (
        train_window.loc[train_window["upc"].isin(products)]
        .groupby("upc")["descrip"]
        .first()
        .to_dict()
    )
    stores = (
        train_window.loc[train_window["upc"].isin(products)]
        .groupby("store")["move"]
        .sum()
        .nlargest(n_stores)
        .index.astype(int)
        .tolist()
    )
    cohort = research_window.loc[
        research_window["upc"].isin(products) & research_window["store"].isin(stores)
    ].copy()

    train = cohort.loc[cohort["week"] <= cutoff].copy()
    panel_median = train.groupby(["upc", "store"])["unit_price"].median().rename("panel_median")
    cohort = cohort.join(panel_median, on=["upc", "store"])
    cohort["relative_price"] = cohort["unit_price"] / cohort["panel_median"]
    cohort = cohort.loc[cohort["relative_price"].between(0.5, 2.0)].copy()

    product_map = {value: index for index, value in enumerate(products)}
    store_map = {value: index for index, value in enumerate(stores)}
    cohort["product_idx"] = cohort["upc"].map(product_map).astype(int)
    cohort["store_idx"] = cohort["store"].map(store_map).astype(int)
    cohort["log_units"] = np.log(cohort["move"])
    cohort["log_price"] = np.log(cohort["unit_price"])
    train_price_mean = (
        cohort.loc[cohort["week"] <= cutoff]
        .groupby(["upc", "store"])["log_price"]
        .mean()
        .rename("train_log_price_mean")
    )
    cohort = cohort.join(train_price_mean, on=["upc", "store"])
    cohort["centered_log_price"] = cohort["log_price"] - cohort["train_log_price_mean"]
    cohort["season_sin"] = np.sin(2 * np.pi * cohort["week"] / 52.0)
    cohort["season_cos"] = np.cos(2 * np.pi * cohort["week"] / 52.0)
    cohort["trend"] = (cohort["week"] - history_start) / history_weeks
    sale = cohort["sale"].fillna("").astype(str).str.strip().str.upper()
    promo_map = {code: index for index, code in enumerate(PROMOTION_CODES)}
    cohort["promo_idx"] = sale.map(promo_map).fillna(-1).astype(int)

    train = cohort.loc[cohort["week"] <= cutoff].reset_index(drop=True)
    holdout = cohort.loc[cohort["week"] > cutoff].reset_index(drop=True)
    return train, holdout, products, stores, product_names


def run_hierarchical_model(
    category: str,
    n_products: int = 8,
    n_stores: int = 12,
    history_weeks: int = 156,
    holdout_weeks: int = 26,
    draws: int = 500,
    tune: int = 500,
    cores: int = 1,
    prior_profile: str = "regularized",
    seed: int = 20260719,
) -> tuple[Path, Path, Path]:
    if prior_profile not in PRIOR_PROFILES:
        raise ValueError(
            f"Unknown prior profile {prior_profile!r}; choose from {sorted(PRIOR_PROFILES)}"
        )
    priors = PRIOR_PROFILES[prior_profile]
    train, holdout, products, stores, product_names = _prepare_cohort(
        category, n_products, n_stores, history_weeks, holdout_weeks
    )
    coords = {
        "product": [str(value) for value in products],
        "store": [str(value) for value in stores],
        "promotion": list(PROMOTION_CODES),
        "obs": np.arange(len(train)),
    }

    with pm.Model(coords=coords):
        product_idx = pm.Data("product_idx", train["product_idx"].to_numpy(), dims="obs")
        store_idx = pm.Data("store_idx", train["store_idx"].to_numpy(), dims="obs")
        centered_log_price = pm.Data(
            "centered_log_price", train["centered_log_price"].to_numpy(), dims="obs"
        )
        promo_idx = train["promo_idx"].to_numpy()
        promo_design = np.column_stack([promo_idx == index for index in range(4)]).astype(float)
        promotion_design_data = pm.Data("promotion_design", promo_design, dims=("obs", "promotion"))
        season_sin_data = pm.Data("season_sin_data", train["season_sin"].to_numpy(), dims="obs")
        season_cos_data = pm.Data("season_cos_data", train["season_cos"].to_numpy(), dims="obs")
        trend_data = pm.Data("trend_data", train["trend"].to_numpy(), dims="obs")

        alpha = pm.Normal("alpha", mu=2.5, sigma=1.5)
        sigma_product = pm.HalfNormal("sigma_product", sigma=1.0)
        sigma_store = pm.HalfNormal("sigma_store", sigma=1.0)
        product_raw = pm.Normal("product_raw", 0, 1, dims="product")
        store_raw = pm.Normal("store_raw", 0, 1, dims="store")
        product_intercept = pm.Deterministic(
            "product_intercept", product_raw * sigma_product, dims="product"
        )
        store_intercept = pm.Deterministic("store_intercept", store_raw * sigma_store, dims="store")

        mean_elasticity = pm.Normal(
            "mean_elasticity",
            mu=priors["mean_elasticity_mu"],
            sigma=priors["mean_elasticity_sigma"],
        )
        sigma_elasticity = pm.HalfNormal("sigma_elasticity", sigma=priors["sigma_elasticity"])
        elasticity_raw = pm.Normal("elasticity_raw", 0, 1, dims="product")
        elasticity_product = pm.Deterministic(
            "elasticity_product",
            mean_elasticity + elasticity_raw * sigma_elasticity,
            dims="product",
        )
        promo_effect = pm.Normal(
            "promo_effect", mu=0.0, sigma=priors["promotion_sigma"], dims="promotion"
        )
        seasonal_sin = pm.Normal("seasonal_sin", 0, 0.5)
        seasonal_cos = pm.Normal("seasonal_cos", 0, 0.5)
        trend_effect = pm.Normal("trend_effect", 0, 0.5)
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Exponential("nu_minus_two", 1 / 15) + 2

        mu = (
            alpha
            + product_intercept[product_idx]
            + store_intercept[store_idx]
            + elasticity_product[product_idx] * centered_log_price
            + pm.math.dot(promotion_design_data, promo_effect)
            + seasonal_sin * season_sin_data
            + seasonal_cos * season_cos_data
            + trend_effect * trend_data
        )
        pm.StudentT("log_units", nu=nu, mu=mu, sigma=sigma, observed=train["log_units"], dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=2,
            cores=cores,
            target_accept=0.92,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )

    suffix = "" if prior_profile == "regularized" else f"_{prior_profile}"
    posterior_path = ARTIFACT_DIR / f"{category}_hierarchical{suffix}_posterior.nc"
    idata.to_netcdf(posterior_path)

    posterior_elasticity = idata.posterior["elasticity_product"]
    elasticity_rows = []
    for index, upc in enumerate(products):
        samples = posterior_elasticity.isel(product=index).to_numpy().ravel()
        elasticity_rows.append(
            {
                "upc": upc,
                "description": product_names.get(upc, ""),
                "posterior_mean": float(samples.mean()),
                "posterior_sd": float(samples.std()),
                "hdi_5pct": float(np.quantile(samples, 0.05)),
                "hdi_95pct": float(np.quantile(samples, 0.95)),
                "probability_elastic_below_minus_one": float((samples < -1).mean()),
            }
        )
    elasticity_path = ARTIFACT_DIR / f"{category}_product_elasticities{suffix}.csv"
    pd.DataFrame(elasticity_rows).to_csv(elasticity_path, index=False)

    holdout_mean, holdout_low, holdout_high = _predict_conditional_mean_log_units(idata, holdout)
    holdout_output = holdout[
        ["upc", "store", "week", "unit_price", "move", "sale", "log_units"]
    ].copy()
    holdout_output["conditional_mean_log_units"] = holdout_mean
    holdout_output["conditional_mean_log_units_p05"] = holdout_low
    holdout_output["conditional_mean_log_units_p95"] = holdout_high
    holdout_path = ARTIFACT_DIR / f"{category}_hierarchical{suffix}_holdout.csv.gz"
    holdout_output.to_csv(holdout_path, index=False, compression="gzip")

    diagnostics = az.summary(
        idata,
        var_names=["mean_elasticity", "sigma_elasticity", "promo_effect"],
        kind="diagnostics",
    )
    divergences = int(idata.sample_stats["diverging"].sum())
    mean_elasticity_samples = idata.posterior["mean_elasticity"].to_numpy().ravel()
    promo_samples = idata.posterior["promo_effect"]
    promo_summary = {}
    for index, code in enumerate(PROMOTION_CODES):
        samples = promo_samples.isel(promotion=index).to_numpy().ravel()
        observations = int((train["promo_idx"] == index).sum())
        promo_summary[code] = {
            "observations": observations,
            "empirically_supported": observations >= 30,
            "posterior_mean_log_effect": float(samples.mean()),
            "posterior_mean_conditional_unit_difference_pct": float(100 * np.expm1(samples.mean())),
            "hdi_5pct": float(np.quantile(samples, 0.05)),
            "hdi_95pct": float(np.quantile(samples, 0.95)),
        }

    holdout_errors = holdout["log_units"].to_numpy() - holdout_mean
    summary = {
        "category": category,
        "prior_profile": prior_profile,
        "prior_parameters": priors,
        "model": "robust hierarchical Bayesian positive-sales price-response model",
        "research_cohort": {
            "products": n_products,
            "stores": n_stores,
            "history_weeks": history_weeks,
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "selection": "Highest-volume products and stores within the pre-holdout research window.",
        },
        "interpretation_guardrail": (
            "Partial-pooled conditional associations, not causal elasticities. "
            "The source contains no positive-price zero-sales observations. Promotion effects "
            "with fewer than 30 cohort observations are prior-dominated and not interpreted."
        ),
        "mean_elasticity": {
            "posterior_mean": float(mean_elasticity_samples.mean()),
            "hdi_5pct": float(np.quantile(mean_elasticity_samples, 0.05)),
            "hdi_95pct": float(np.quantile(mean_elasticity_samples, 0.95)),
        },
        "promotion_effects": promo_summary,
        "holdout_log_mae": float(np.mean(np.abs(holdout_errors))),
        "holdout_log_rmse": float(np.sqrt(np.mean(holdout_errors**2))),
        "diagnostics": {
            "divergences": divergences,
            "max_rhat": float(diagnostics["r_hat"].max()),
            "min_bulk_ess": float(diagnostics["ess_bulk"].min()),
        },
    }
    summary_path = ARTIFACT_DIR / f"{category}_hierarchical{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path, elasticity_path, posterior_path
