from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR


def run_isolated_promotion_event_study(
    category: str,
    promotion_code: str = "S",
    window: int = 4,
) -> tuple[Path, Path]:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    frame = pd.read_csv(
        panel_path,
        usecols=[
            "upc",
            "store",
            "week",
            "move",
            "unit_price",
            "gross_margin_dollars",
            "sale",
        ],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )
    frame["sale_code"] = frame["sale"].fillna("").astype(str).str.strip().str.upper()
    frame["log_units"] = np.log(frame["move"])
    frame["log_price"] = np.log(frame["unit_price"])

    events = frame.loc[frame["sale_code"].eq(promotion_code), ["upc", "store", "week"]].copy()
    events = events.rename(columns={"week": "event_week"}).reset_index(drop=True)
    events["event_id"] = np.arange(len(events))
    relative = pd.DataFrame({"relative_week": np.arange(-window, window + 1)})
    grid = events.merge(relative, how="cross")
    grid["week"] = grid["event_week"] + grid["relative_week"]
    lookup = frame.drop(columns="sale").rename(columns={"week": "lookup_week"})
    grid = grid.merge(
        lookup,
        left_on=["upc", "store", "week"],
        right_on=["upc", "store", "lookup_week"],
        how="left",
        validate="many_to_one",
    )

    complete = grid.groupby("event_id")["move"].count().eq(2 * window + 1)
    correct_center = (
        grid.loc[grid["relative_week"].eq(0)].set_index("event_id")["sale_code"].eq(promotion_code)
    )
    quiet_neighbors = (
        grid.loc[grid["relative_week"].ne(0)]
        .groupby("event_id")["sale_code"]
        .apply(lambda values: values.fillna("").eq("").all())
    )
    valid_ids = complete.index[complete & correct_center & quiet_neighbors]
    study = grid.loc[grid["event_id"].isin(valid_ids)].copy()
    if study.empty:
        raise ValueError(f"No isolated {promotion_code} events satisfy the window requirements")

    baseline = (
        study.loc[study["relative_week"].isin([-4, -3, -2])]
        .groupby("event_id")
        .agg(
            baseline_log_units=("log_units", "mean"),
            baseline_log_price=("log_price", "mean"),
            baseline_margin=("gross_margin_dollars", "mean"),
        )
    )
    study = study.join(baseline, on="event_id")
    study["delta_log_units"] = study["log_units"] - study["baseline_log_units"]
    study["delta_log_price"] = study["log_price"] - study["baseline_log_price"]
    study["delta_margin_dollars"] = study["gross_margin_dollars"] - study["baseline_margin"]

    def aggregate(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        unit_mean = group["delta_log_units"].mean()
        unit_se = group["delta_log_units"].std(ddof=1) / np.sqrt(n)
        price_mean = group["delta_log_price"].mean()
        margin_mean = group["delta_margin_dollars"].mean()
        margin_se = group["delta_margin_dollars"].std(ddof=1) / np.sqrt(n)
        return pd.Series(
            {
                "events": n,
                "mean_delta_log_units": unit_mean,
                "approx_unit_difference_pct": 100 * np.expm1(unit_mean),
                "unit_difference_pct_low": 100 * np.expm1(unit_mean - 1.96 * unit_se),
                "unit_difference_pct_high": 100 * np.expm1(unit_mean + 1.96 * unit_se),
                "mean_delta_log_price": price_mean,
                "approx_price_difference_pct": 100 * np.expm1(price_mean),
                "mean_delta_margin_dollars": margin_mean,
                "margin_difference_low": margin_mean - 1.96 * margin_se,
                "margin_difference_high": margin_mean + 1.96 * margin_se,
            }
        )

    profile = (
        study.groupby("relative_week", sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    profile_path = ARTIFACT_DIR / f"{category}_{promotion_code}_isolated_event_profile.csv"
    profile.to_csv(profile_path, index=False)

    def row(relative_week: int) -> pd.Series:
        return profile.loc[profile["relative_week"].eq(relative_week)].iloc[0]

    post_1_2 = profile.loc[profile["relative_week"].isin([1, 2]), "mean_delta_log_units"].mean()
    post_3_4 = profile.loc[profile["relative_week"].isin([3, 4]), "mean_delta_log_units"].mean()
    summary = {
        "category": category,
        "promotion_code": promotion_code,
        "design": (
            "Isolated one-week promotion events with four observed non-promotion weeks on each side; "
            "outcomes normalized to weeks -4 through -2."
        ),
        "event_count": int(study["event_id"].nunique()),
        "question": (
            "Does the observed promotion coincide with a temporary unit spike, and is it followed "
            "by a demand dip consistent with purchase timing or stockpiling?"
        ),
        "causal_status": (
            "Descriptive event study, not causal. Event timing may be selected and promotion flags are incomplete."
        ),
        "week_minus_1_pretrend_unit_difference_pct": float(row(-1)["approx_unit_difference_pct"]),
        "event_week_unit_difference_pct": float(row(0)["approx_unit_difference_pct"]),
        "event_week_price_difference_pct": float(row(0)["approx_price_difference_pct"]),
        "event_week_margin_difference_dollars": float(row(0)["mean_delta_margin_dollars"]),
        "post_weeks_1_2_unit_difference_pct": float(100 * np.expm1(post_1_2)),
        "post_weeks_3_4_unit_difference_pct": float(100 * np.expm1(post_3_4)),
    }
    summary_path = ARTIFACT_DIR / f"{category}_{promotion_code}_event_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path, profile_path


def run_price_promotion_end_study(
    category: str,
    discount_threshold: float = 0.10,
    n_products: int = 50,
    pre_weeks: tuple[int, ...] = (-4, -3, -2),
    post_window: int = 8,
    max_episode_weeks: int = 8,
) -> tuple[Path, Path]:
    """Describe demand after price-derived promotion episodes end.

    The source sale flag is incomplete, so episodes use price depth relative to each
    store-product median. This remains descriptive because median price is only a proxy
    for regular price and promotion timing is endogenous.
    """
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    frame = pd.read_csv(
        panel_path,
        usecols=["upc", "store", "week", "move", "unit_price"],
        dtype={"store": "int32", "week": "int16", "move": "float32"},
        low_memory=False,
    )
    frame = frame.loc[(frame["move"] > 0) & (frame["unit_price"] > 0)].copy()
    products = frame.groupby("upc")["move"].sum().nlargest(n_products).index
    frame = frame.loc[frame["upc"].isin(products)].copy()
    frame["panel_id"] = frame["store"].astype(str) + ":" + frame["upc"].astype(str)
    regular_price = frame.groupby("panel_id")["unit_price"].transform("median")
    frame["discount_depth"] = np.maximum(0.0, 1 - frame["unit_price"] / regular_price)
    frame["price_promotion"] = frame["discount_depth"] >= discount_threshold
    frame["log_units"] = np.log(frame["move"])
    frame = frame.sort_values(["panel_id", "week"]).reset_index(drop=True)

    group = frame.groupby("panel_id", sort=False)
    previous_promotion = group["price_promotion"].shift(1).eq(True)
    previous_week = group["week"].shift(1)
    frame["episode_start"] = frame["price_promotion"] & (
        ~previous_promotion | (frame["week"] - previous_week != 1)
    )
    frame["episode_number"] = frame.groupby("panel_id", sort=False)["episode_start"].cumsum()
    promotional = frame.loc[frame["price_promotion"]].copy()
    episodes = promotional.groupby(["panel_id", "episode_number"], as_index=False).agg(
        upc=("upc", "first"),
        store=("store", "first"),
        start_week=("week", "min"),
        end_week=("week", "max"),
        episode_weeks=("week", "size"),
        mean_depth=("discount_depth", "mean"),
    )
    episodes = episodes.loc[episodes["episode_weeks"].between(1, max_episode_weeks)].copy()
    episodes["episode_id"] = np.arange(len(episodes))

    lookup = frame[["panel_id", "week", "log_units", "price_promotion"]].copy()
    baseline_offsets = pd.DataFrame({"relative_week": list(pre_weeks)})
    baseline_grid = episodes[["episode_id", "panel_id", "start_week"]].merge(
        baseline_offsets, how="cross"
    )
    baseline_grid["week"] = baseline_grid["start_week"] + baseline_grid["relative_week"]
    baseline_grid = baseline_grid.merge(
        lookup, on=["panel_id", "week"], how="left", validate="many_to_one"
    )
    baseline_valid = baseline_grid.groupby("episode_id").agg(
        observed=("log_units", "count"),
        any_promotion=(
            "price_promotion",
            lambda values: bool(values.isna().any() or values.eq(True).any()),
        ),
    )
    baseline_valid = baseline_valid.index[
        baseline_valid["observed"].eq(len(pre_weeks)) & ~baseline_valid["any_promotion"]
    ]
    baseline = (
        baseline_grid.loc[baseline_grid["episode_id"].isin(baseline_valid)]
        .groupby("episode_id")["log_units"]
        .mean()
        .rename("baseline_log_units")
    )

    post_offsets = pd.DataFrame({"weeks_after_end": np.arange(1, post_window + 1)})
    post = episodes[["episode_id", "panel_id", "end_week"]].merge(post_offsets, how="cross")
    post["week"] = post["end_week"] + post["weeks_after_end"]
    post = post.merge(lookup, on=["panel_id", "week"], how="left", validate="many_to_one")
    post_valid = post.groupby("episode_id").agg(
        observed=("log_units", "count"),
        any_promotion=(
            "price_promotion",
            lambda values: bool(values.isna().any() or values.eq(True).any()),
        ),
    )
    post_valid = post_valid.index[
        post_valid["observed"].eq(post_window) & ~post_valid["any_promotion"]
    ]
    valid_ids = baseline.index.intersection(post_valid)
    study = post.loc[post["episode_id"].isin(valid_ids)].join(baseline, on="episode_id")
    study["delta_log_units"] = study["log_units"] - study["baseline_log_units"]
    if study.empty:
        raise ValueError("No complete uncontaminated promotion-end episodes were found")

    def aggregate(group: pd.DataFrame) -> pd.Series:
        estimate = float(group["delta_log_units"].mean())
        centered = group["delta_log_units"] - estimate
        cluster_sums = centered.groupby(group["panel_id"]).sum().to_numpy()
        clusters = len(cluster_sums)
        variance = float(np.sum(cluster_sums**2) / len(group) ** 2)
        if clusters > 1:
            variance *= clusters / (clusters - 1)
        standard_error = float(np.sqrt(variance))
        return pd.Series(
            {
                "episodes": int(group["episode_id"].nunique()),
                "panels": int(group["panel_id"].nunique()),
                "mean_delta_log_units": estimate,
                "unit_difference_pct": float(100 * np.expm1(estimate)),
                "unit_difference_pct_low": float(100 * np.expm1(estimate - 1.96 * standard_error)),
                "unit_difference_pct_high": float(100 * np.expm1(estimate + 1.96 * standard_error)),
            }
        )

    profile = (
        study.groupby("weeks_after_end", sort=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )
    prefix = f"{category}_price_promotion_end"
    profile_path = ARTIFACT_DIR / f"{prefix}_profile.csv"
    profile.to_csv(profile_path, index=False)
    early = profile.loc[profile["weeks_after_end"].between(1, 4), "mean_delta_log_units"].mean()
    late = profile.loc[profile["weeks_after_end"].between(5, 8), "mean_delta_log_units"].mean()
    selected_episodes = episodes.loc[episodes["episode_id"].isin(valid_ids)]
    summary = {
        "category": category,
        "design": (
            "Descriptive promotion-end study for 1-8 week price-derived episodes, requiring "
            "three clean pre-start reference weeks and eight observed non-promotion weeks after end."
        ),
        "episode_definition": (
            f"Per-unit price at least {discount_threshold:.0%} below the store-product median; "
            "the median is a regular-price proxy."
        ),
        "episodes": int(len(selected_episodes)),
        "panels": int(selected_episodes["panel_id"].nunique()),
        "mean_episode_weeks": float(selected_episodes["episode_weeks"].mean()),
        "mean_episode_depth": float(selected_episodes["mean_depth"].mean()),
        "post_weeks_1_4_unit_difference_pct": float(100 * np.expm1(early)),
        "post_weeks_5_8_unit_difference_pct": float(100 * np.expm1(late)),
        "causal_status": (
            "Descriptive only. Price-derived treatment can be misclassified, the panel omits "
            "zero-sales offered-price weeks, and promotion timing is endogenous."
        ),
    }
    summary_path = ARTIFACT_DIR / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path, profile_path
