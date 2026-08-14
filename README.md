# Retail Promotion Decision Engine

## Question

Which products should a retailer promote, at what discount, and in which stores
when the objective is incremental contribution margin rather than unit lift or
revenue alone?

## Data

The analysis uses the University of Chicago Booth Kilts Center Dominick's Finer
Foods scanner data. Store-product-week panels are built from the source files
with explicit validation for keys, prices, quantities, source quality, and
bundle-price semantics. Raw data is not included.

The source has material limitations: promotion coding is incomplete, historical
price and merchandising decisions were not randomized, zero-sales demand and
stockouts are not observed cleanly, and accounting acquisition cost is not a
current replacement-cost feed.

## Evidence chain

1. A validated DuckDB mart establishes store-product-week grain and provenance.
2. Fixed-effects and hierarchical Bayesian models estimate conditional price
   response with temporal holdouts and partial pooling.
3. A doubly robust promotion analysis is subjected to overlap, balance, placebo,
   and negative-control tests before it can influence optimization.
4. A mixed-integer optimizer compares revenue and contribution-margin scenarios
   under capacity and downside-risk constraints.
5. Release gates block recommendations when causal identification, calibration,
   monitoring, or economic inputs are inadequate.

## Findings

| Evidence | Result | Decision implication |
| --- | --- | --- |
| Bayesian holdout uncertainty | 81.9% coverage for a nominal 90% interval | Model is overconfident |
| Split-conformal recalibration | 85.4% coverage on a later window | Improvement, still below target |
| Historical shadow replay | Twelve of thirteen weeks trigger at least one block | Aggregate performance hides unstable weeks |
| Strict propensity overlap | 29.1% of rows | Historical promotion effect is not decision-grade |
| Post-weighting balance | Maximum absolute SMD 0.224 | Residual imbalance remains material |
| Revenue scenario | +$1,159 revenue and −$481 contribution margin | Unit/revenue lift does not imply profitability |
| Downside-screened margin scenario | Zero actions at all tested capacities | No promotion is the robust base-case decision |

The large adjusted historical promotion estimate is withheld because overlap,
balance, a future-treatment placebo, and a past-outcome negative control fail.
Numerical model convergence does not override failed identification or
under-covered uncertainty.

## Decision

Do not automate the historical promotion policy. The current system is useful
as a refusal mechanism and experiment-design tool: it identifies which evidence
is missing, prevents an invalid causal estimate from reaching the optimizer, and
keeps “no action” available when margin risk is unfavorable.

The next evidence should come from a stratified randomized promotion experiment
with explicit discount depth, display and feature status, offered prices on
zero-sales weeks, inventory and stockouts, supplier funding, and current
replacement cost. Incremental units and contribution margin should be
pre-registered as co-primary outcomes.

## Limitations

- Price-response coefficients are conditional associations, not causal
  elasticities.
- The recorded promotion flag is incomplete and unlabeled weeks are not proven
  controls.
- The demand model conditions on positive sales and cannot identify censored
  demand or stockouts.
- Optimizer economics use historical accounting-cost proxies.
- Substitution, basket effects, residual demand shocks, and future regime changes
  are outside the current scenario model.
- A containerized, monitored service is not evidence that the policy is safe to
  deploy.

The [executive decision memo](docs/executive_decision_memo.md) explains the
blocked recommendation. The [model card](docs/model_card.md) contains the full
evaluation, causal gate, scenario assumptions, and release criteria. The
[external validation note](docs/external_validation.md) separates local findings
from published evidence.

## Data use

[Dominick's data](https://www.chicagobooth.edu/research/kilts/research-data/dominicks)
is distributed by the Kilts Center for academic research. Users must follow the
provider's terms and acknowledge the source. Raw files and derived row-level
data are not redistributed here.
