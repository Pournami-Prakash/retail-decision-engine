# Executive Decision Memo

## Decision

Do not deploy the observational promotion estimate or use it to automate promotion selection. Retain the current engine as a research and experiment-design tool. The defensible next decision is to run a controlled promotion test with complete operational and economic measurement.

## What the analysis answered

The project asked a business question that unit-lift reporting misses: which promotions create incremental contribution margin after accounting for price response, uncertainty, and operating constraints?

Historical cereal promotions coincide with a dramatic event-week unit increase of 415.5% and a 32.2% price decline. Average event-week gross margin also rises by $12.44 in the descriptive cohort. But the pre-event week is already 7.2% above baseline, warning that promotion timing is selected rather than exchangeable.

The associational scenario optimizer makes the commercial tradeoff concrete. Under base capacity, maximizing revenue selects 12 actions and models approximately $1,159 of incremental revenue while reducing contribution margin by approximately $481. A downside-screened margin objective selects zero under conservative, base, and expanded capacity. The non-obvious conclusion is that visible demand lift and revenue growth can rationally coexist with a “do not promote” margin decision.

The zero-action result applies only to price-cut response under the stated proxy economics. It does not establish that real bundled promotions are unprofitable. Unmeasured display, feature, loyalty, basket, or other merchandising could add genuine lift, but the observed promotion-code associations also contain endogenous targeting. Their causal contribution and direction are therefore unknown and deliberately excluded.

## Why the causal estimate is rejected

The doubly robust model reports a large promotion effect, but its falsification evidence fails:

- Strict propensity overlap covers only 29.1% of rows.
- Residual weighted imbalance reaches 0.224, above the 0.10 threshold.
- A future-treatment placebo is statistically different from zero.
- A redesigned past-outcome negative control, using only covariates measured before that outcome, is statistically different from zero.

These are not cosmetic diagnostics. Together they indicate residual selection, timing structure, or measurement problems large enough that the point estimate should not drive spending. The decision gate therefore withholds it from the optimizer.

A separate price-derived promotion-end study found units still +3.3% in weeks 1–4 and +1.9% in weeks 5–8 for its tightly filtered cohort. That does not prove there is no stockpiling: treatment is inferred from median price and broader external work reports a modest post-promotion dip under a different episode definition. The actionable conclusion is to measure payback in the experiment, not assume either result generalizes.

## Model-health finding

The Bayesian price-response model converges numerically, but its untouched-holdout uncertainty is too narrow. The nominal 90% interval covers only 81.9% of observations and the predictive distribution understates observed dispersion. Prior sensitivity is acceptable at the product level but not at the global level. This is useful evidence about what the next model must fix; it is not production clearance.

Chronological split-conformal recalibration improves nominal-90% coverage to 85.4% on a strictly later evaluation window, but still misses the target. The original and recalibrated results are both retained; recalibration does not rewrite model history.

## Recommended experiment

Randomize eligible store-product-weeks into no-promotion and a small number of explicit discount depths, stratified by store, product, and baseline velocity. Pre-register unit sales and incremental contribution margin as co-primary outcomes. Capture display/feature status, offered price including zero-sales weeks, inventory and stockouts, supplier funding, and current replacement cost. Use holdout stores or staggered randomization to measure spillovers and demand pull-forward.

This creates four practical benefits: it prevents margin-destructive promotions from being justified by unit lift, estimates the supplier funding needed to make a candidate profitable, directs scarce test cells toward decisions with the highest value of information, and creates model-health alerts before recommendations reach merchandising.

Funding is commercially material in the current scenario: no candidate clears the downside screen without funding, while $0.10 per promoted unit allows 12 modeled actions with approximately $14 aggregate funded margin. That is a negotiation threshold to test with replacement-cost data—not a guaranteed return. A planning calculation for a 20% unit lift, 13 weeks, and intraclass correlation 0.05 requires approximately 43 stores per arm; a pilot must estimate the actual correlation and margin variance.

Promote a policy to production only after it passes overlap, balance, falsification, predictive-calibration, and downside-margin thresholds. Until then, use the dashboard to inspect scenarios and design the test—not to claim causal ROI.
