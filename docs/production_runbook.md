# Production Runbook

## Scope

This runbook describes how the decision service should fail safely. It does not
declare the historical Dominick's policy production-ready.

## Release sequence

1. Rebuild source manifests and validate the store-item-time grain.
2. Fit into a versioned artifact directory and record code, data, configuration,
   and artifact hashes.
3. Evaluate the untouched temporal holdout and the later conformal evaluation
   window. Do not replace the original uncertainty results with recalibrated ones.
4. Run the 13-week historical shadow replay and inspect weekly coverage, price
   PSI, residual bias, and promotion-rate context before considering live shadowing.
5. Run known-truth causal implementation tests, then the real-data overlap,
   balance, placebo, and negative-control gate.
6. Require current replacement cost, supplier funding, inventory availability,
   explicit offer mechanics, and zero-sales offered-price rows.
7. Validate the randomized panel against
   `contracts/promotion_experiment_observation.schema.json`, preserve the allocation
   file, and run the block-adjusted, store-clustered intent-to-treat analysis.
8. Admit only arms whose lower 95% confidence bound for incremental contribution
   margin is positive and whose stockout upper bound remains within tolerance.
9. Shadow-score a live batch and compare feature PSI, interval coverage, residual
   bias, recommendation rate, and no-action rate with the registered baseline.
10. Run the unified release gate, then require merchandising, finance, data-science,
   and responsible-pricing approval.

## Evidence commands

```bash
.venv/bin/retail-decision experiment-analyze \
  --category cereal \
  --input path/to/completed_promotion_experiment_observations.csv
.venv/bin/retail-decision release-gate --category cereal
```

The repository ships a header-only collection template. Running it proves that the
system blocks missing evidence; it is not an executed experiment or sample result.

## Automatic blocks

- missing or changed unregistered artifacts;
- invalid request schema or out-of-scope store-item-discount;
- PSI at least 0.25, missing live-batch monitoring, or calibration below threshold;
- causal admission failure or unavailable economic inputs;
- replacement cost greater than or equal to the funded promoted price;
- unavailable inventory or an unmet store/product capacity constraint.

## Rollback

The service defaults to no recommendation. Rollback means restoring the last
registered artifact bundle, verifying its hashes, and keeping automated decisions
disabled until a shadow batch passes. The observational scenario dashboard remains
available for analysis but cannot bypass the decision gate.

## Monitoring ownership

- Data engineering: freshness, contracts, row counts, keys, missingness, PSI.
- Data science: calibration, residuals, causal diagnostics, treatment support.
- Merchandising and finance: replacement cost, funding, capacity, realized margin.
- Platform: latency, errors, artifact integrity, audit retention, rollback.
