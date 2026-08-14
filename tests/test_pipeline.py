import unittest

import numpy as np
import pandas as pd

from retail_decision_engine.baseline import _fit_single_regressor
from retail_decision_engine.benchmark import _clustered_ols
from retail_decision_engine.pipeline import _normalize_columns, _require_columns
from retail_decision_engine.optimizer import _break_even_elasticity, _solve_candidates
from retail_decision_engine.monitoring import (
    _metric_status,
    drift_status,
    population_stability_index,
)
from retail_decision_engine.operational import score_decision_request
from retail_decision_engine.calibration import _coverage, _split_conformal_temporal
from retail_decision_engine.causal import (
    NEGATIVE_CONTROL_FEATURES,
    _clustered_mean_interval,
    _prior_expanding_mean,
)
from retail_decision_engine.causal_validation import _synthetic_panel
from retail_decision_engine.experiment import (
    _itt_by_arm,
    _stores_per_arm,
    validate_experiment_frame,
)


class PipelineTests(unittest.TestCase):
    def test_column_normalization(self) -> None:
        frame = pd.DataFrame(columns=[" UPC ", "Store", "PRICE"])
        normalized = _normalize_columns(frame)
        self.assertEqual(list(normalized.columns), ["upc", "store", "price"])

    def test_required_columns_fails_with_clear_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns: price"):
            _require_columns(pd.DataFrame(columns=["upc"]), {"upc", "price"}, "movement")

    def test_single_regressor_recovers_known_coefficient(self) -> None:
        x = np.array([-2.0, -1.0, 1.0, 2.0])
        y = -1.5 * x
        coefficient, standard_error = _fit_single_regressor(x, y)
        self.assertAlmostEqual(coefficient, -1.5)
        self.assertAlmostEqual(standard_error, 0.0)

    def test_single_regressor_rejects_zero_variation(self) -> None:
        with self.assertRaisesRegex(ValueError, "no within-group variation"):
            _fit_single_regressor(np.zeros(4), np.arange(4.0))

    def test_clustered_ols_recovers_known_coefficients(self) -> None:
        x = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
        beta = np.array([-2.0, 0.5])
        fitted, standard_errors = _clustered_ols(x, x @ beta, pd.Series(["a", "a", "b", "b"]))
        np.testing.assert_allclose(fitted, beta)
        np.testing.assert_allclose(standard_errors, np.zeros(2), atol=1e-12)

    def test_break_even_elasticity_respects_margin_economics(self) -> None:
        threshold = _break_even_elasticity(0.20, 0.05)
        self.assertLess(threshold, -5.0)
        self.assertTrue(np.isnan(_break_even_elasticity(0.20, 0.20)))

    def test_optimizer_enforces_one_discount_per_store_product(self) -> None:
        candidates = pd.DataFrame(
            {
                "store": [1, 1, 1],
                "upc": [100, 100, 200],
                "incremental_margin_mean": [10.0, 8.0, 7.0],
                "incremental_margin_p10": [5.0, 4.0, 3.0],
                "incremental_revenue_mean": [12.0, 9.0, 8.0],
            }
        )
        result = _solve_candidates(
            candidates,
            "incremental_margin_mean",
            "test",
            max_promotions=2,
            max_per_store=2,
            max_per_product=2,
            require_positive_risk_floor=True,
        )
        self.assertEqual(len(result.selected), 2)
        self.assertEqual(result.selected["incremental_margin_mean"].sum(), 17.0)

    def test_predictive_coverage_uses_central_interval(self) -> None:
        actual = np.array([0.0, 3.0])
        draws = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
        result = _coverage(actual, draws, 0.80)
        self.assertEqual(result["nominal"], 0.80)
        self.assertEqual(result["empirical"], 0.50)

    def test_prior_expanding_mean_excludes_current_row(self) -> None:
        result = _prior_expanding_mean(pd.Series([2.0, 4.0, 9.0]))
        self.assertTrue(np.isnan(result.iloc[0]))
        np.testing.assert_allclose(result.iloc[1:].to_numpy(), [2.0, 3.0])

    def test_clustered_interval_reports_mean_and_cluster_count(self) -> None:
        result = _clustered_mean_interval(
            np.array([1.0, 1.0, 3.0, 3.0]), pd.Series(["a", "a", "b", "b"])
        )
        self.assertEqual(result["estimate_log_points"], 2.0)
        self.assertEqual(result["clusters"], 2)
        self.assertEqual(result["rows"], 4)
        self.assertGreater(result["clustered_standard_error"], 0)

    def test_negative_control_does_not_use_its_outcome_as_feature(self) -> None:
        self.assertNotIn("lag2_log_units", NEGATIVE_CONTROL_FEATURES)
        self.assertNotIn("lag1_log_units", NEGATIVE_CONTROL_FEATURES)
        self.assertTrue(all(feature.startswith("nc_") for feature in NEGATIVE_CONTROL_FEATURES))

    def test_cluster_bootstrap_coverage_is_bounded(self) -> None:
        actual = np.array([0.0, 0.0, 3.0, 3.0])
        draws = np.tile(np.array([-1.0, 0.0, 1.0]), (4, 1))
        result = _coverage(actual, draws, 0.80, clusters=np.array(["a", "a", "b", "b"]))
        self.assertLessEqual(result["cluster_bootstrap_low"], result["empirical"])
        self.assertGreaterEqual(result["cluster_bootstrap_high"], result["empirical"])

    def test_temporal_conformal_never_evaluates_on_calibration_weeks(self) -> None:
        weeks = np.repeat(np.arange(1, 9), 2)
        actual = np.linspace(0, 1.5, len(weeks))
        center = np.zeros(len(weeks))
        clusters = np.tile(["a", "b"], 8)
        result, low, high = _split_conformal_temporal(actual, center, weeks, clusters)
        self.assertEqual(result["split_week"], 5)
        self.assertTrue(np.isnan(low[weeks < 5]).all())
        self.assertTrue(np.isfinite(high[weeks >= 5]).all())

    def test_drift_monitor_passes_identical_distributions(self) -> None:
        reference = np.linspace(0, 1, 1000)
        psi = population_stability_index(reference, reference.copy())
        self.assertAlmostEqual(psi, 0.0)
        self.assertEqual(drift_status(psi), "pass")

    def test_coverage_monitor_distinguishes_warning_from_block(self) -> None:
        self.assertEqual(_metric_status(0.91, 0.90, 0.85), "pass")
        self.assertEqual(_metric_status(0.87, 0.90, 0.85), "warn")
        self.assertEqual(_metric_status(0.80, 0.90, 0.85), "block")

    def test_decision_contract_fails_closed_on_missing_economics(self) -> None:
        result = score_decision_request(
            {"category": "cereal", "store": 8, "upc": 1, "discount": 0.05}
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("missing:replacement_unit_cost", result["reasons"])

    def test_known_truth_panel_contains_both_treatment_arms(self) -> None:
        frame = _synthetic_panel("randomized", seed=7, panels=10, weeks=8)
        self.assertEqual(set(frame["treatment"].unique()), {0, 1})
        self.assertTrue(set(NEGATIVE_CONTROL_FEATURES).issubset(frame.columns))

    def test_experiment_power_requires_more_stores_for_smaller_effect(self) -> None:
        small = _stores_per_arm(0.8, np.log1p(0.10), 13, 0.05)
        large = _stores_per_arm(0.8, np.log1p(0.30), 13, 0.05)
        self.assertGreater(small, large)

    def test_experiment_intake_rejects_missing_operational_fields(self) -> None:
        result = validate_experiment_frame(
            pd.DataFrame({"experiment_id": ["test"], "assigned_treatment": [1]})
        )
        self.assertFalse(result["intake_passed"])
        self.assertIn("replacement_unit_cost", result["missing_columns"])

    def test_randomized_itt_recovers_known_arm_effect(self) -> None:
        frame = pd.DataFrame(
            {
                "assigned_treatment": [0, 1] * 20,
                "assigned_arm": ["control", "discount_10"] * 20,
                "randomization_block": [f"wave_{index // 10}" for index in range(40)],
                "store": np.repeat(np.arange(1, 11), 4),
            }
        )
        frame["units_sold"] = 5.0 + 2.0 * frame["assigned_treatment"]
        result = _itt_by_arm(frame, "units_sold")
        self.assertEqual(result[0]["arm"], "discount_10")
        self.assertAlmostEqual(result[0]["estimate"], 2.0)


if __name__ == "__main__":
    unittest.main()
