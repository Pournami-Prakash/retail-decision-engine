from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .bayesian import PROMOTION_CODES, _posterior_samples
from .calibration import _coverage
from .config import ARTIFACT_DIR, PROCESSED_DIR


DIAGNOSTIC_VARIABLES = [
    "global_elasticity",
    "sigma_category_elasticity",
    "category_elasticity",
    "sigma_product_elasticity",
    "product_elasticity",
    "sigma_store_elasticity",
    "store_elasticity_offset",
    "sigma_product_intercept",
    "sigma_store",
    "product_intercept",
    "store_intercept",
    "promo_effect",
    "seasonal_sin",
    "seasonal_cos",
    "trend_effect",
    "sigma",
    "nu_minus_two",
]


def _sampling_diagnostics(idata: az.InferenceData) -> dict[str, float | int]:
    diagnostics = az.summary(idata, var_names=DIAGNOSTIC_VARIABLES, kind="diagnostics")
    return {
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "max_rhat": float(diagnostics["r_hat"].max()),
        "min_bulk_ess": float(diagnostics["ess_bulk"].min()),
        "min_tail_ess": float(diagnostics["ess_tail"].min()),
        "variables_audited": len(diagnostics),
    }


def _prepare_multicategory(
    categories: list[str],
    products_per_category: int,
    stores_per_category: int,
    history_weeks: int,
    holdout_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[int], np.ndarray]:
    frames = []
    for category in categories:
        path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; download and build the category first")
        frame = pd.read_csv(
            path,
            usecols=["upc", "store", "week", "move", "unit_price", "sale", "descrip"],
            dtype={"store": "int32", "week": "int16", "move": "float32"},
            low_memory=False,
        )
        frame = frame.loc[(frame["move"] > 0) & (frame["unit_price"] > 0)].copy()
        max_week = int(frame["week"].max())
        cutoff = max_week - holdout_weeks
        frame = frame.loc[frame["week"] > cutoff - history_weeks].copy()
        training = frame.loc[frame["week"] <= cutoff]
        products = training.groupby("upc")["move"].sum().nlargest(products_per_category).index
        stores = (
            training.loc[training["upc"].isin(products)]
            .groupby("store")["move"]
            .sum()
            .nlargest(stores_per_category)
            .index
        )
        frame = frame.loc[frame["upc"].isin(products) & frame["store"].isin(stores)].copy()
        frame["category"] = category
        frame["cutoff"] = cutoff
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["product_key"] = combined["category"] + ":" + combined["upc"].astype(str)
    combined["panel_key"] = combined["product_key"] + ":" + combined["store"].astype(str)
    combined["log_units"] = np.log(combined["move"])
    combined["log_price"] = np.log(combined["unit_price"])

    train_mask = combined["week"] <= combined["cutoff"]
    train_price_mean = (
        combined.loc[train_mask].groupby("panel_key")["log_price"].mean().rename("price_mean")
    )
    combined = combined.join(train_price_mean, on="panel_key")
    combined["centered_log_price"] = combined["log_price"] - combined["price_mean"]
    relative_price = combined["unit_price"] / np.exp(combined["price_mean"])
    combined = combined.loc[relative_price.between(0.5, 2.0)].copy()

    category_levels = categories
    product_levels = sorted(combined["product_key"].unique())
    store_levels = sorted(combined["store"].unique().astype(int).tolist())
    category_map = {value: index for index, value in enumerate(category_levels)}
    product_map = {value: index for index, value in enumerate(product_levels)}
    store_map = {value: index for index, value in enumerate(store_levels)}
    combined["category_idx"] = combined["category"].map(category_map).astype(int)
    combined["product_idx"] = combined["product_key"].map(product_map).astype(int)
    combined["store_idx"] = combined["store"].map(store_map).astype(int)
    product_category = (
        combined.drop_duplicates("product_key")
        .set_index("product_key")["category_idx"]
        .reindex(product_levels)
        .astype(int)
        .to_numpy()
    )
    sale = combined["sale"].fillna("").astype(str).str.strip().str.upper()
    promo_map = {code: index for index, code in enumerate(PROMOTION_CODES)}
    combined["promo_idx"] = sale.map(promo_map).fillna(-1).astype(int)
    combined["season_sin"] = np.sin(2 * np.pi * combined["week"] / 52.0)
    combined["season_cos"] = np.cos(2 * np.pi * combined["week"] / 52.0)
    combined["trend"] = combined.groupby("category")["week"].transform(
        lambda values: (values - values.min()) / history_weeks
    )
    train = combined.loc[combined["week"] <= combined["cutoff"]].reset_index(drop=True)
    holdout = combined.loc[combined["week"] > combined["cutoff"]].reset_index(drop=True)
    return train, holdout, category_levels, product_levels, store_levels, product_category


def _multicategory_linear_draws(idata: az.InferenceData, frame: pd.DataFrame) -> np.ndarray:
    alpha = _posterior_samples(idata, "alpha")
    product_intercept = _posterior_samples(idata, "product_intercept")
    store_intercept = _posterior_samples(idata, "store_intercept")
    product_elasticity = _posterior_samples(idata, "product_elasticity")
    store_elasticity_offset = _posterior_samples(idata, "store_elasticity_offset")
    promo_effect = _posterior_samples(idata, "promo_effect")
    seasonal_sin = _posterior_samples(idata, "seasonal_sin")
    seasonal_cos = _posterior_samples(idata, "seasonal_cos")
    trend_effect = _posterior_samples(idata, "trend_effect")
    product_idx = frame["product_idx"].to_numpy()
    store_idx = frame["store_idx"].to_numpy()
    prediction = (
        alpha[None, :]
        + product_intercept[product_idx, :]
        + store_intercept[store_idx, :]
        + (product_elasticity[product_idx, :] + store_elasticity_offset[store_idx, :])
        * frame["centered_log_price"].to_numpy()[:, None]
        + seasonal_sin[None, :] * frame["season_sin"].to_numpy()[:, None]
        + seasonal_cos[None, :] * frame["season_cos"].to_numpy()[:, None]
        + trend_effect[None, :] * frame["trend"].to_numpy()[:, None]
    )
    promo_idx = frame["promo_idx"].to_numpy()
    promoted = promo_idx >= 0
    prediction[promoted] += promo_effect[promo_idx[promoted], :]
    return prediction


def _evaluate_multicategory(
    idata: az.InferenceData,
    holdout: pd.DataFrame,
    label: str,
    seed: int,
) -> tuple[Path, Path]:
    linear_draws = _multicategory_linear_draws(idata, holdout)
    sigma = _posterior_samples(idata, "sigma")
    nu = _posterior_samples(idata, "nu_minus_two") + 2
    rng = np.random.default_rng(seed)
    predictive = (
        linear_draws + rng.standard_t(df=nu[None, :], size=linear_draws.shape) * sigma[None, :]
    )
    actual = holdout["log_units"].to_numpy()
    predicted = predictive.mean(axis=1)
    clusters = holdout["panel_key"].to_numpy()
    coverage = {
        str(int(probability * 100)): _coverage(
            actual, predictive, probability, clusters=clusters, seed=seed
        )
        for probability in (0.50, 0.80, 0.90, 0.95)
    }
    rows = []
    for category, group in holdout.groupby("category", sort=False):
        index = group.index.to_numpy()
        category_actual = actual[index]
        category_predicted = predicted[index]
        category_draws = predictive[index]
        rows.append(
            {
                "category": category,
                "rows": int(len(index)),
                "log_mae": float(np.mean(np.abs(category_actual - category_predicted))),
                "log_rmse": float(np.sqrt(np.mean((category_actual - category_predicted) ** 2))),
                "log_bias_actual_minus_predicted": float(
                    np.mean(category_actual - category_predicted)
                ),
                "coverage_90": float(
                    (
                        (category_actual >= np.quantile(category_draws, 0.05, axis=1))
                        & (category_actual <= np.quantile(category_draws, 0.95, axis=1))
                    ).mean()
                ),
            }
        )
    category_path = ARTIFACT_DIR / f"multicategory_{label}_holdout_by_category.csv"
    pd.DataFrame(rows).to_csv(category_path, index=False)
    evaluation = {
        "target": (
            "conditional temporal holdout scoring with observed prices and promotion codes; "
            "not a future-covariate forecast"
        ),
        "rows": int(len(holdout)),
        "log_mae": float(np.mean(np.abs(actual - predicted))),
        "log_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "log_bias_actual_minus_predicted": float(np.mean(actual - predicted)),
        "predictive_interval_calibration": coverage,
        "by_category_file": str(category_path),
    }
    evaluation_path = ARTIFACT_DIR / f"multicategory_{label}_holdout_evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    return evaluation_path, category_path


def evaluate_multicategory_model(
    categories: list[str],
    products_per_category: int = 4,
    stores_per_category: int = 6,
    history_weeks: int = 120,
    holdout_weeks: int = 26,
    seed: int = 20260722,
) -> tuple[Path, Path, Path]:
    _, holdout, _, _, _, _ = _prepare_multicategory(
        categories,
        products_per_category,
        stores_per_category,
        history_weeks,
        holdout_weeks,
    )
    label = "_".join(categories)
    posterior_path = ARTIFACT_DIR / f"multicategory_{label}_posterior.nc"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing {posterior_path}; fit the multi-category model first")
    idata = az.from_netcdf(posterior_path)
    evaluation_path, category_path = _evaluate_multicategory(idata, holdout, label, seed)
    summary_path = ARTIFACT_DIR / f"multicategory_{label}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["diagnostics"] = _sampling_diagnostics(idata)
    summary["holdout_evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary["interpretation_guardrail"] = (
        "Exploratory positive-sales associations among selected high-volume UPCs. Category means "
        "are not representative causal category rankings. Store effects are partially pooled "
        "intercepts and cross-classified store elasticity offsets."
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path, evaluation_path, category_path


def run_multicategory_model(
    categories: list[str],
    products_per_category: int = 4,
    stores_per_category: int = 6,
    history_weeks: int = 120,
    holdout_weeks: int = 26,
    draws: int = 300,
    tune: int = 300,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.97,
    seed: int = 20260721,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    train, holdout, category_levels, product_levels, store_levels, product_category = (
        _prepare_multicategory(
            categories,
            products_per_category,
            stores_per_category,
            history_weeks,
            holdout_weeks,
        )
    )
    coords = {
        "category": category_levels,
        "product": product_levels,
        "store": [str(value) for value in store_levels],
        "promotion": list(PROMOTION_CODES),
        "obs": np.arange(len(train)),
    }
    promo_idx = train["promo_idx"].to_numpy()
    promo_design = np.column_stack([promo_idx == index for index in range(4)]).astype(float)
    with pm.Model(coords=coords):
        product_idx = pm.Data("product_idx", train["product_idx"], dims="obs")
        store_idx = pm.Data("store_idx", train["store_idx"], dims="obs")
        product_category_data = pm.Data("product_category", product_category, dims="product")
        price = pm.Data("centered_log_price", train["centered_log_price"], dims="obs")
        promo = pm.Data("promotion_design", promo_design, dims=("obs", "promotion"))
        season_sin_data = pm.Data("season_sin_data", train["season_sin"], dims="obs")
        season_cos_data = pm.Data("season_cos_data", train["season_cos"], dims="obs")
        trend_data = pm.Data("trend_data", train["trend"], dims="obs")

        alpha = pm.Normal("alpha", 2.5, 1.5)
        sigma_product_intercept = pm.HalfNormal("sigma_product_intercept", 1.0)
        sigma_store = pm.HalfNormal("sigma_store", 1.0)
        product_intercept = pm.Deterministic(
            "product_intercept",
            pm.Normal("product_intercept_raw", 0, 1, dims="product") * sigma_product_intercept,
            dims="product",
        )
        store_intercept = pm.Deterministic(
            "store_intercept",
            pm.Normal("store_intercept_raw", 0, 1, dims="store") * sigma_store,
            dims="store",
        )

        global_elasticity = pm.Normal("global_elasticity", -1.5, 1.0)
        sigma_category_elasticity = pm.HalfNormal("sigma_category_elasticity", 0.75)
        category_elasticity = pm.Deterministic(
            "category_elasticity",
            global_elasticity
            + pm.Normal("category_elasticity_raw", 0, 1, dims="category")
            * sigma_category_elasticity,
            dims="category",
        )
        sigma_product_elasticity = pm.HalfNormal("sigma_product_elasticity", 0.75)
        product_elasticity = pm.Deterministic(
            "product_elasticity",
            category_elasticity[product_category_data]
            + pm.Normal("product_elasticity_raw", 0, 1, dims="product") * sigma_product_elasticity,
            dims="product",
        )
        sigma_store_elasticity = pm.HalfNormal("sigma_store_elasticity", 0.5)
        store_elasticity_offset = pm.Deterministic(
            "store_elasticity_offset",
            pm.Normal("store_elasticity_raw", 0, 1, dims="store") * sigma_store_elasticity,
            dims="store",
        )
        promo_effect = pm.Normal("promo_effect", 0, 1, dims="promotion")
        seasonal_sin = pm.Normal("seasonal_sin", 0, 0.5)
        seasonal_cos = pm.Normal("seasonal_cos", 0, 0.5)
        trend_effect = pm.Normal("trend_effect", 0, 0.5)
        sigma = pm.HalfNormal("sigma", 1.0)
        nu = pm.Exponential("nu_minus_two", 1 / 15) + 2
        mu = (
            alpha
            + product_intercept[product_idx]
            + store_intercept[store_idx]
            + (product_elasticity[product_idx] + store_elasticity_offset[store_idx]) * price
            + pm.math.dot(promo, promo_effect)
            + seasonal_sin * season_sin_data
            + seasonal_cos * season_cos_data
            + trend_effect * trend_data
        )
        pm.StudentT("log_units", nu=nu, mu=mu, sigma=sigma, observed=train["log_units"], dims="obs")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            return_inferencedata=True,
        )

    label = "_".join(categories)
    posterior_path = ARTIFACT_DIR / f"multicategory_{label}_posterior.nc"
    idata.to_netcdf(posterior_path)
    category_rows = []
    category_samples = idata.posterior["category_elasticity"]
    for index, category in enumerate(category_levels):
        samples = category_samples.isel(category=index).to_numpy().ravel()
        category_rows.append(
            {
                "category": category,
                "posterior_mean": float(samples.mean()),
                "p05": float(np.quantile(samples, 0.05)),
                "p95": float(np.quantile(samples, 0.95)),
            }
        )
    category_path = ARTIFACT_DIR / f"multicategory_{label}_category_elasticities.csv"
    pd.DataFrame(category_rows).to_csv(category_path, index=False)

    product_rows = []
    product_samples = idata.posterior["product_elasticity"]
    for index, key in enumerate(product_levels):
        samples = product_samples.isel(product=index).to_numpy().ravel()
        product_rows.append(
            {
                "product_key": key,
                "category": key.split(":", 1)[0],
                "posterior_mean": float(samples.mean()),
                "p05": float(np.quantile(samples, 0.05)),
                "p95": float(np.quantile(samples, 0.95)),
            }
        )
    product_path = ARTIFACT_DIR / f"multicategory_{label}_product_elasticities.csv"
    pd.DataFrame(product_rows).to_csv(product_path, index=False)

    store_rows = []
    store_offset_samples = idata.posterior["store_elasticity_offset"]
    for index, store in enumerate(store_levels):
        samples = store_offset_samples.isel(store=index).to_numpy().ravel()
        store_rows.append(
            {
                "store": store,
                "elasticity_offset_mean": float(samples.mean()),
                "p05": float(np.quantile(samples, 0.05)),
                "p95": float(np.quantile(samples, 0.95)),
            }
        )
    store_path = ARTIFACT_DIR / f"multicategory_{label}_store_elasticity_offsets.csv"
    pd.DataFrame(store_rows).to_csv(store_path, index=False)

    summary = {
        "categories": categories,
        "model": "robust hierarchical Bayesian positive-sales price-response model",
        "sampling": {
            "chains": chains,
            "draws_per_chain": draws,
            "tuning_steps_per_chain": tune,
            "target_accept": target_accept,
            "seed": seed,
        },
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "products": len(product_levels),
        "stores": len(store_levels),
        "hierarchy": (
            "cross-classified global -> category -> product elasticity slopes plus partially "
            "pooled store elasticity offsets, store intercepts, and product intercepts"
        ),
        "diagnostics": _sampling_diagnostics(idata),
    }
    summary_path = ARTIFACT_DIR / f"multicategory_{label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    evaluation_path, evaluation_category_path = _evaluate_multicategory(
        idata, holdout, label, seed + 1
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["holdout_evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
    summary["interpretation_guardrail"] = (
        "Exploratory positive-sales associations among selected high-volume UPCs. Category means "
        "are not representative causal category rankings. Store effects are partially pooled "
        "intercepts and cross-classified store elasticity offsets."
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return (
        summary_path,
        category_path,
        product_path,
        store_path,
        evaluation_path,
        evaluation_category_path,
    )
