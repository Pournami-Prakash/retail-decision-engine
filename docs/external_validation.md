# External Validation Note

Checked 20 July 2026.

## Source semantics

The [official Dominick's data manual](https://www.chicagobooth.edu/research/kilts/research-data/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx) confirms the transformations and guardrails used here:

- individual-unit price is `PRICE / QTY` and revenue is `PRICE * MOVE / QTY`;
- `PROFIT` is a gross-margin percentage;
- the underlying cost is average acquisition cost, not current replacement cost;
- `OK = 0` identifies suspect observations;
- a populated sale code indicates a promotion, but a blank code does not prove that no promotion occurred.

These details are why the optimizer is a scenario tool and why “recorded S versus unlabeled” is the precise observational treatment contrast.

## Independent result comparison

A March 2026 [non-peer-reviewed cereal preprint](https://www.preprints.org/manuscript/202603.1966) provides a useful independent reasonableness check, not ground truth. It reports the same 6,602,582 raw cereal rows, 4.64 million rows after its own trimming, a +426.95% promotion-start spike, and a +6.80% week −1 lead. This project finds +415.5% and +7.2% for isolated recorded one-week simple-sale events. The close scale supports the data construction and the descriptive event pattern.

The studies diverge after promotions end. The preprint reports a roughly 2–5% dip using broader price-promotion episodes; this project's more restrictive price-derived cohort remains +3.3% in weeks 1–4 and +1.9% in weeks 5–8. That is not evidence that one analysis is “right.” Episode thresholds, reference windows, sample restrictions, and treatment misclassification change the target population. The business implication is to measure pull-forward and pantry loading explicitly in a controlled test.

## What external checking does not validate

Agreement on row counts and descriptive spikes does not validate causal identification, future forecasts, accounting-cost economics, or deployment. Those claims remain blocked by the local overlap, balance, falsification, calibration, and data-availability checks.
