# Model Card: Retail Promotion Decision Engine

## Intended use

This research system estimates conditional price response, audits recorded promotions, and compares constrained retail decisions. It is intended for analyst review, research demonstration, and hypothesis generation. It is not a production pricing system and must not autonomously set prices.

## Data

The pipeline uses the University of Chicago Booth Kilts Center Dominick's Finer Foods scanner data, not Kaggle. It creates store-product-week panels from historical transaction aggregates and preserves source provenance with SHA-256 manifests and validation reports.

A DuckDB analytical mart implements a star schema with product, store, and retail-week dimensions at an explicit store-product-week fact grain. Automated checks cover fact-key uniqueness, orphaned dimensions, positive measures, and finite revenue and margin values.

The source has important limitations: promotion coding is incomplete; positive prices appear only with positive sales; stockouts are not identified; accounting acquisition cost is not necessarily current replacement cost; and historical merchandising assignment was not randomized.

The official manual documents sale codes B, C, and S. The raw files also contain rare undocumented codes; the model treats the common undocumented code as a separate nuisance indicator and does not give it a business interpretation.

## Models and estimands

- A store-product and week fixed-effects benchmark estimates within-panel conditional price association.
- A robust hierarchical Bayesian model partially pools product elasticities and store/product intercepts while controlling for promotion codes, seasonality, and trend. Student-t residuals limit the influence of extreme observations.
- A multi-category hierarchy pools global, category, and product price response.
- A store-product-panel-cross-fitted augmented inverse propensity-weighted estimator contrasts recorded simple-sale weeks with unlabeled weeks on log positive-sale units. Contiguous calendar-time blocks provide a sensitivity analysis. Because unlabeled does not mean untreated, this is a candidate estimand—not an identified causal effect.
- A mixed-integer program selects promotions under portfolio, store, product, and downside-margin constraints.

The price-response coefficients are conditional associations, not causal elasticities. Doubly robust estimation protects against misspecification of either the propensity or outcome nuisance model under its assumptions; it does not solve unmeasured confounding.

## Evaluation

The cereal Bayesian model used 11,409 training rows and an untouched 2,388-row, 26-week temporal holdout. Sampling produced zero divergences, maximum R-hat 1.01, and minimum bulk ESS 351.

Temporal-holdout error was 0.577 MAE and 0.853 RMSE on the log-unit scale. Predictive intervals were under-covered: empirical coverage was 43.4%, 71.6%, 81.9%, and 88.1% for nominal 50%, 80%, 90%, and 95% intervals. The observed log-unit standard deviation was 0.998, above the posterior-predictive 5th–95th percentile range of 0.700–0.779. The reference-only PIT uniformity test also rejects calibration, but its iid p-value is anti-conservative for dependent panel observations. Release decisions use coverage with store-product cluster-bootstrap intervals. The model is overconfident and is not cleared for automated decisions.

A sequential shadow replay makes the temporal failure operational rather than hypothetical. Using only the preceding 13 weeks to update intervals, the final 13 untouched weeks contain 1,150 scored rows and reach 87.7% coverage against a 90% target. Twelve of thirteen weeks trigger at least one coverage, residual-bias, or price-PSI block. Week 397 is the highest-drift batch: promotion prevalence rises to 27.3%, price PSI reaches 1.86, and mean log-scale bias reaches +0.71. The replay demonstrates that monitoring works and identifies the regime shift; it does not convert a historical backtest into live production evidence.

The basic hierarchical holdout file labels parameter-uncertainty bands as `conditional_mean_log_units_p05/p95`. They are not predictive intervals. Full posterior-predictive bands—including Student-t residual noise—are emitted only by the calibration workflow, preventing the two uncertainty targets from being confused.

A chronological split-conformal layer is calibrated on the first half of the holdout and evaluated only on the later half. Its nominal-90% interval covers 85.4% of later observations, with a store-product cluster-bootstrap interval of 82.2%–88.1%. This improves uncertainty without reaching target. Exchangeability is not assumed for the time-dependent panel, so the observed later-window result—not a theoretical conformal guarantee—controls the gate.

Across regularized, skeptical, and weak prior profiles, mean category elasticity ranged from -1.850 to -2.177. That 0.327 range fails the project's global stability threshold of 0.25. Individual product means were much more stable, with a maximum range of 0.027.

The cross-category model uses 7,283 training and 1,722 temporal-holdout rows across 12 products and 12 selected stores. It now cross-classifies global/category/product price-response slopes with partially pooled store response offsets, plus store and product intercepts. Four sequential chains with 800 tuning and 500 retained draws each produced zero divergences, maximum R-hat 1.01, minimum bulk ESS 496, and minimum tail ESS 603 across 66 audited parameters. Holdout log MAE is 0.608 and nominal-90% coverage is 80.7% overall: cereal 89.4%, canned soup 83.8%, and soft drinks 69.4%. Store offsets describe selected-store heterogeneity shared across modeled categories; they do not establish causal local elasticities. The selected high-volume category contrasts remain exploratory rather than representative category rankings.

## Causal admission gate

The promotion estimate is withheld from optimization. Only 29.1% of observations were strictly inside the 0.05–0.95 overlap region, maximum post-weighting standardized mean difference was 0.224, and both the future-treatment placebo and past-outcome negative control excluded zero. For the redesigned negative control, the outcome is at t−2, all adjustment variables end at t−3 or earlier, and continuous observed history from t−6 through t is required. Its estimate is −0.0311 log points with a 95% interval from −0.0392 to −0.0230. Panel cross-fitting estimates +580% while contiguous-time-block cross-fitting estimates +521%; stability of a large adjusted association does not repair failed identification.

Known-truth simulations separately validate estimator plumbing: absolute error is 0.024 under randomized assignment and 0.032 with measured confounding. When an omitted variable drives treatment and outcome, error rises to 0.624. This is expected and intentionally demonstrates why the historical estimate remains blocked.

## Decision policy

The optimizer is explicitly a scenario tool. It bootstraps recent baseline volume, price, and accounting-margin inputs alongside posterior price-response draws, but excludes residual shocks, substitution, stockouts, supplier funding, future covariates, and true replacement cost. Under base capacity, the revenue objective selects 12 actions with modeled +$1,159 revenue and −$481 contribution margin. The downside-screened margin objective selects zero under conservative (6), base (12), and expanded (24) action limits. “No action” is a valid robust scenario result, not a solver failure.

That zero is conditional on a **price-cut-only** response. It is not evidence that fully bundled promotions are generally unprofitable. Display, feature, loyalty targeting, basket attachment, or other merchandising may create genuine incremental lift, while the observed promotion-code coefficients can also contain endogenous targeting and demand selection. Because those components are not identified, excluding them can either omit genuine lift or prevent selection-contaminated lift from being overstated; the direction of the missing causal contribution is unknown.

Supplier funding is now explicit. Zero funding produces no downside-screened actions; a $0.10-per-promoted-unit scenario produces 12 modeled actions and +$14 aggregate funded margin, while $0.25 produces 12 and +$127. These remain negotiation scenarios using accounting-cost proxies and associational demand response—not causal funding recommendations.

## Monitoring and release criteria

A production candidate should remain blocked unless all of the following hold:

- 90% temporal-holdout predictive interval coverage is close to nominal and no worse than 85%.
- PIT and residual diagnostics show no material systematic calibration failure.
- Maximum R-hat is at most 1.01, minimum bulk ESS is at least 400, and divergences are zero.
- Strict propensity overlap covers at least 80% of the decision cohort.
- Every post-weighting covariate has absolute standardized mean difference at most 0.10.
- Pre-specified placebo and negative-control confidence intervals include zero.
- Prior sensitivity stays within business-defined tolerances.
- Inventory, stockouts, explicit promotion mechanics, supplier funding, and replacement-cost data are available.

## Known risks

Selection into promotion, incomplete treatments, censored zero demand, price endogeneity, changing retail regimes, and proxy economics can all bias recommendations. Results should be reviewed by data science, merchandising, finance, and responsible-pricing stakeholders before any experiment.
