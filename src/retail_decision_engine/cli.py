from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import run_elasticity_baseline
from .benchmark import run_two_way_benchmark
from .bayesian import run_hierarchical_model
from .optimizer import run_promotion_optimizer
from .event_study import run_isolated_promotion_event_study, run_price_promotion_end_study
from .calibration import run_posterior_predictive_calibration
from .sensitivity import run_prior_sensitivity
from .causal import run_doubly_robust_promotion_analysis
from .causal_validation import run_causal_implementation_validation
from .multicategory import evaluate_multicategory_model, run_multicategory_model
from .gating import run_causal_decision_gate
from .sql_mart import build_sql_mart
from .operational import run_operational_readiness
from .monitoring import run_historical_shadow_replay
from .experiment import analyze_randomized_experiment, run_experiment_plan
from .release import run_release_gate
from .service import serve_decisions
from .download import download_categories
from .pipeline import build_category_panel


def _categories(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-decision")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download first-party source files")
    download.add_argument("--categories", type=_categories, required=True)
    download.add_argument("--force", action="store_true")

    build = commands.add_parser("build", help="Validate and build analytical panels")
    build.add_argument("--categories", type=_categories, required=True)

    baseline = commands.add_parser("baseline", help="Fit a fixed-effects elasticity baseline")
    baseline.add_argument("--category", required=True)
    baseline.add_argument("--holdout-weeks", type=int, default=26)

    benchmark = commands.add_parser(
        "benchmark", help="Fit two-way fixed effects with promotion controls"
    )
    benchmark.add_argument("--category", required=True)
    benchmark.add_argument("--holdout-weeks", type=int, default=26)

    bayesian = commands.add_parser("bayesian", help="Fit hierarchical Bayesian elasticity model")
    bayesian.add_argument("--category", required=True)
    bayesian.add_argument("--products", type=int, default=8)
    bayesian.add_argument("--stores", type=int, default=12)
    bayesian.add_argument("--history-weeks", type=int, default=156)
    bayesian.add_argument("--holdout-weeks", type=int, default=26)
    bayesian.add_argument("--draws", type=int, default=500)
    bayesian.add_argument("--tune", type=int, default=500)
    bayesian.add_argument("--cores", type=int, default=1)
    bayesian.add_argument(
        "--prior-profile", choices=["regularized", "skeptical", "weak"], default="regularized"
    )

    optimize = commands.add_parser("optimize", help="Optimize discount portfolios under risk")
    optimize.add_argument("--category", required=True)
    optimize.add_argument("--products", type=int, default=8)
    optimize.add_argument("--stores", type=int, default=12)
    optimize.add_argument("--history-weeks", type=int, default=156)
    optimize.add_argument("--holdout-weeks", type=int, default=26)

    event_study = commands.add_parser("event-study", help="Analyze isolated promotion windows")
    event_study.add_argument("--category", required=True)
    event_study.add_argument("--promotion-code", default="S")

    payback_study = commands.add_parser(
        "payback-study", help="Analyze demand after price-derived promotion episodes end"
    )
    payback_study.add_argument("--category", required=True)
    payback_study.add_argument("--discount-threshold", type=float, default=0.10)
    payback_study.add_argument("--products", type=int, default=50)

    calibrate = commands.add_parser("calibrate", help="Run full posterior predictive calibration")
    calibrate.add_argument("--category", required=True)
    calibrate.add_argument("--products", type=int, default=8)
    calibrate.add_argument("--stores", type=int, default=12)
    calibrate.add_argument("--history-weeks", type=int, default=156)
    calibrate.add_argument("--holdout-weeks", type=int, default=26)

    sensitivity = commands.add_parser("sensitivity", help="Fit and compare alternative priors")
    sensitivity.add_argument("--category", required=True)
    sensitivity.add_argument("--products", type=int, default=8)
    sensitivity.add_argument("--stores", type=int, default=12)
    sensitivity.add_argument("--history-weeks", type=int, default=156)
    sensitivity.add_argument("--holdout-weeks", type=int, default=26)
    sensitivity.add_argument("--draws", type=int, default=300)
    sensitivity.add_argument("--tune", type=int, default=300)

    causal = commands.add_parser("causal", help="Run doubly robust promotion analysis")
    causal.add_argument("--category", required=True)
    causal.add_argument("--history-weeks", type=int, default=180)
    causal.add_argument("--products", type=int, default=30)

    commands.add_parser(
        "causal-validate", help="Validate AIPW implementation on known-truth panels"
    )

    multicategory = commands.add_parser("multicategory", help="Fit global/category/product pooling")
    multicategory.add_argument("--categories", type=_categories, required=True)
    multicategory.add_argument("--products-per-category", type=int, default=4)
    multicategory.add_argument("--stores-per-category", type=int, default=6)
    multicategory.add_argument("--history-weeks", type=int, default=120)
    multicategory.add_argument("--holdout-weeks", type=int, default=26)
    multicategory.add_argument("--draws", type=int, default=300)
    multicategory.add_argument("--tune", type=int, default=300)
    multicategory.add_argument("--chains", type=int, default=4)
    multicategory.add_argument("--cores", type=int, default=1)
    multicategory.add_argument("--target-accept", type=float, default=0.97)

    multicategory_evaluate = commands.add_parser(
        "multicategory-evaluate", help="Evaluate the multi-category model on its temporal holdout"
    )
    multicategory_evaluate.add_argument("--categories", type=_categories, required=True)
    multicategory_evaluate.add_argument("--products-per-category", type=int, default=4)
    multicategory_evaluate.add_argument("--stores-per-category", type=int, default=6)
    multicategory_evaluate.add_argument("--history-weeks", type=int, default=120)
    multicategory_evaluate.add_argument("--holdout-weeks", type=int, default=26)

    gate = commands.add_parser("gate", help="Gate causal estimates before optimization")
    gate.add_argument("--category", required=True)

    sql_mart = commands.add_parser("sql-mart", help="Build a validated dimensional DuckDB mart")
    sql_mart.add_argument("--category", required=True)
    commands.add_parser(
        "operational-check", help="Build artifact lineage and fail-closed readiness report"
    )
    experiment = commands.add_parser(
        "experiment-plan", help="Build a power-sensitive randomized promotion test plan"
    )
    experiment.add_argument("--category", default="cereal")
    experiment_analyze = commands.add_parser(
        "experiment-analyze", help="Validate and analyze a randomized promotion panel"
    )
    experiment_analyze.add_argument("--input", type=Path, required=True)
    experiment_analyze.add_argument("--category", default="cereal")
    experiment_analyze.add_argument("--stockout-tolerance", type=float, default=0.02)
    release_gate = commands.add_parser(
        "release-gate", help="Evaluate randomized, economic, calibration, and monitoring evidence"
    )
    release_gate.add_argument("--category", default="cereal")
    release_gate.add_argument("--coverage-target", type=float, default=0.90)
    shadow = commands.add_parser(
        "shadow-replay", help="Replay real historical holdout batches through monitoring"
    )
    shadow.add_argument("--category", default="cereal")
    shadow.add_argument("--lookback-weeks", type=int, default=13)
    serve = commands.add_parser("serve", help="Run the fail-closed decision HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download":
        print(download_categories(args.categories, force=args.force))
    elif args.command == "build":
        for category in args.categories:
            panel, report = build_category_panel(category)
            print(panel)
            print(report)
    elif args.command == "baseline":
        result, predictions = run_elasticity_baseline(args.category, args.holdout_weeks)
        print(result)
        print(predictions)
    elif args.command == "benchmark":
        print(run_two_way_benchmark(args.category, args.holdout_weeks))
    elif args.command == "bayesian":
        for path in run_hierarchical_model(
            args.category,
            n_products=args.products,
            n_stores=args.stores,
            history_weeks=args.history_weeks,
            holdout_weeks=args.holdout_weeks,
            draws=args.draws,
            tune=args.tune,
            cores=args.cores,
            prior_profile=args.prior_profile,
        ):
            print(path)
    elif args.command == "optimize":
        for path in run_promotion_optimizer(
            args.category,
            n_products=args.products,
            n_stores=args.stores,
            history_weeks=args.history_weeks,
            holdout_weeks=args.holdout_weeks,
        ):
            print(path)
    elif args.command == "event-study":
        for path in run_isolated_promotion_event_study(
            args.category, promotion_code=args.promotion_code.upper()
        ):
            print(path)
    elif args.command == "payback-study":
        for path in run_price_promotion_end_study(
            args.category,
            discount_threshold=args.discount_threshold,
            n_products=args.products,
        ):
            print(path)
    elif args.command == "calibrate":
        for path in run_posterior_predictive_calibration(
            args.category,
            n_products=args.products,
            n_stores=args.stores,
            history_weeks=args.history_weeks,
            holdout_weeks=args.holdout_weeks,
        ):
            print(path)
    elif args.command == "sensitivity":
        print(
            run_prior_sensitivity(
                args.category,
                n_products=args.products,
                n_stores=args.stores,
                history_weeks=args.history_weeks,
                holdout_weeks=args.holdout_weeks,
                draws=args.draws,
                tune=args.tune,
            )
        )
    elif args.command == "causal":
        for path in run_doubly_robust_promotion_analysis(
            args.category,
            history_weeks=args.history_weeks,
            n_products=args.products,
        ):
            print(path)
    elif args.command == "causal-validate":
        print(run_causal_implementation_validation())
    elif args.command == "multicategory":
        for path in run_multicategory_model(
            args.categories,
            products_per_category=args.products_per_category,
            stores_per_category=args.stores_per_category,
            history_weeks=args.history_weeks,
            holdout_weeks=args.holdout_weeks,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
        ):
            print(path)
    elif args.command == "multicategory-evaluate":
        for path in evaluate_multicategory_model(
            args.categories,
            products_per_category=args.products_per_category,
            stores_per_category=args.stores_per_category,
            history_weeks=args.history_weeks,
            holdout_weeks=args.holdout_weeks,
        ):
            print(path)
    elif args.command == "gate":
        print(run_causal_decision_gate(args.category))
    elif args.command == "sql-mart":
        for path in build_sql_mart(args.category):
            print(path)
    elif args.command == "operational-check":
        for path in run_operational_readiness():
            print(path)
    elif args.command == "experiment-plan":
        print(run_experiment_plan(args.category))
    elif args.command == "experiment-analyze":
        for path in analyze_randomized_experiment(
            args.input, args.category, stockout_tolerance=args.stockout_tolerance
        ):
            print(path)
    elif args.command == "release-gate":
        print(run_release_gate(args.category, coverage_target=args.coverage_target))
    elif args.command == "shadow-replay":
        for path in run_historical_shadow_replay(
            args.category, lookback_weeks=args.lookback_weeks
        ):
            print(path)
    elif args.command == "serve":
        serve_decisions(args.host, args.port)


if __name__ == "__main__":
    main()
