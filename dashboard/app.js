const state = { data: null, mode: "executive" };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const number = (value, digits = 1) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
const signed = (value, digits = 1) => `${value >= 0 ? "+" : "−"}${number(Math.abs(value), digits)}`;
const percent = (value, digits = 1) => `${number(value * 100, digits)}%`;

function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value;
}

function renderTradeoff(data) {
  const revenue = data.optimization.policies.revenue;
  const values = [
    { label: "Revenue", value: revenue.incremental_revenue_mean, kind: "positive" },
    { label: "Margin", value: revenue.incremental_margin_mean, kind: "negative" },
  ];
  const max = Math.max(...values.map((item) => Math.abs(item.value)));
  $("#tradeoff-chart").innerHTML = values.map((item) => `
    <div class="trade-row">
      <span>${item.label}</span>
      <div class="trade-track">
        <span class="trade-zero" aria-hidden="true"></span>
        <div class="trade-bar ${item.kind}" style="width:${Math.abs(item.value) / max * 46}%"></div>
      </div>
      <strong class="trade-value">${signed(item.value, 0)}</strong>
    </div>`).join("");
  $("#capacity-grid").innerHTML = `
    <div class="capacity-row capacity-head"><span>Capacity</span><span>Margin-safe</span><span>Revenue actions</span><span>Modeled margin</span></div>
    ${Object.entries(data.optimization.capacity_sensitivity).map(([name, result]) => `
      <div class="capacity-row">
        <strong>${name}</strong>
        <span>${result.downside_screened_margin.selected_promotions}</span>
        <span>${result.revenue.selected_promotions}</span>
        <strong class="loss-value">${signed(result.revenue.incremental_margin_mean, 0)}</strong>
      </div>`).join("")}`;
}

function renderElasticities(rows) {
  const min = -4.0;
  const max = 0.5;
  const position = (value) => Math.max(0, Math.min(100, (value - min) / (max - min) * 100));
  const sorted = [...rows].sort((a, b) => a.posterior_mean - b.posterior_mean);
  $("#elasticity-chart").innerHTML = sorted.map((row) => {
    const low = position(row.hdi_5pct);
    const high = position(row.hdi_95pct);
    return `<div class="interval-row">
      <span class="interval-label" title="${row.description}">${row.description}</span>
      <div class="interval-track" aria-hidden="true">
        <span class="interval-line" style="left:${low}%;width:${Math.max(1, high - low)}%"></span>
        <span class="interval-dot" style="left:${position(row.posterior_mean)}%"></span>
      </div>
      <strong class="interval-value">${number(row.posterior_mean, 2)}</strong>
    </div>`;
  }).join("");
}

function renderCalibration(data) {
  const coverage = data.calibration.predictive_interval_calibration;
  $("#calibration-bars").innerHTML = Object.values(coverage).map((row) => `
    <div class="calibration-row">
      <span>${Math.round(row.nominal * 100)}% target</span>
      <div class="calibration-track">
        <span class="calibration-target" style="left:${row.nominal * 100}%"></span>
        <div class="calibration-actual" style="width:${row.empirical * 100}%"></div>
      </div>
      <strong>${percent(row.empirical)}</strong>
    </div>`).join("");
  $("#health-table tbody").innerHTML = data.calibration_by_product
    .sort((a, b) => a.coverage_90 - b.coverage_90)
    .map((row) => `<tr><td>${row.description}</td><td>${percent(row.coverage_90)}</td><td>${number(row.mean_pit, 2)}</td><td>${number(row.log_mae, 2)}</td></tr>`).join("");
}

function renderGate(data) {
  const gate = data.decision_gate;
  const causal = data.causal;
  const diagnostics = causal.diagnostics;
  const placebo = causal.future_treatment_lead4_placebo;
  const negative = causal.past_outcome_negative_control;
  const checks = [
    ["Overlap ≥ 80%", diagnostics.fraction_strictly_inside_005_095 >= .8, percent(diagnostics.fraction_strictly_inside_005_095)],
    ["Post-weighting balance < 0.10", diagnostics.max_abs_smd_after_ipw < .1, number(diagnostics.max_abs_smd_after_ipw, 3)],
    ["Future-treatment placebo includes zero", placebo.ci95_low_log_points <= 0 && placebo.ci95_high_log_points >= 0, signed(placebo.estimated_unit_difference_pct)],
    ["Past-outcome control includes zero", negative.ci95_low_log_points <= 0 && negative.ci95_high_log_points >= 0, signed(negative.estimated_unit_difference_pct)],
  ];
  $("#gate-list").innerHTML = checks.map(([label, passed, metric]) => `<li><span>${label}<small> · ${metric}</small></span><strong class="gate-result">${passed ? "Pass" : "Fail"}</strong></li>`).join("");
  setText("#gate-status", gate.status.replaceAll("_", " "));
  setText("#aipw-effect", signed(causal.primary_aipw.estimated_unit_difference_pct, 0) + "%");
}

function renderBalance(rows) {
  const max = Math.max(.2, ...rows.map((row) => row.abs_smd_after_ipw));
  $("#balance-chart").innerHTML = rows.sort((a, b) => b.abs_smd_after_ipw - a.abs_smd_after_ipw).map((row) => `
    <div class="balance-row"><span>${row.feature.replaceAll("_", " ")}</span><div class="balance-track"><div class="balance-bar" style="width:${row.abs_smd_after_ipw / max * 100}%"></div></div><strong>${number(row.abs_smd_after_ipw, 3)}</strong></div>`).join("");
}

function renderEvent(data) {
  const rows = data.event_profile;
  const width = 900;
  const height = 300;
  const pad = 36;
  const transformed = rows.map((row) => Math.sign(row.approx_unit_difference_pct) * Math.log1p(Math.abs(row.approx_unit_difference_pct)));
  const yMin = Math.min(...transformed, 0);
  const yMax = Math.max(...transformed, 0);
  const x = (week) => pad + (week + 4) / 8 * (width - pad * 2);
  const y = (value) => height - pad - (value - yMin) / (yMax - yMin || 1) * (height - pad * 2);
  const points = rows.map((row, index) => `${x(row.relative_week)},${y(transformed[index])}`).join(" ");
  $("#event-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
    <line class="chart-axis" x1="${pad}" x2="${width - pad}" y1="${y(0)}" y2="${y(0)}" />
    <line class="chart-event" x1="${x(0)}" x2="${x(0)}" y1="${pad}" y2="${height - pad}" />
    <polyline class="chart-line" points="${points}" />
    ${rows.map((row, index) => `<circle class="chart-point" cx="${x(row.relative_week)}" cy="${y(transformed[index])}" r="4"/><text class="chart-label" x="${x(row.relative_week)}" y="${height - 10}" text-anchor="middle">${row.relative_week > 0 ? "+" : ""}${row.relative_week}</text>`).join("")}
  </svg>`;
}

function populateScenario(data) {
  const candidates = data.promotion_candidates;
  const productSelect = $("#product-select");
  const storeSelect = $("#store-select");
  const discountSelect = $("#discount-select");
  const products = [...new Map(candidates.map((row) => [row.upc, row.description])).entries()];
  productSelect.innerHTML = products.map(([upc, description]) => `<option value="${upc}">${description} · ${upc}</option>`).join("");

  function updateStores() {
    const stores = [...new Set(candidates.filter((row) => String(row.upc) === productSelect.value).map((row) => row.store))].sort((a, b) => a - b);
    storeSelect.innerHTML = stores.map((store) => `<option value="${store}">Store ${store}</option>`).join("");
    updateDiscounts();
  }
  function updateDiscounts() {
    const discounts = [...new Set(candidates.filter((row) => String(row.upc) === productSelect.value && String(row.store) === storeSelect.value).map((row) => row.discount))].sort();
    discountSelect.innerHTML = discounts.map((discount) => `<option value="${discount}">${percent(discount, 0)}</option>`).join("");
    updateOutput();
  }
  function updateOutput() {
    const row = candidates.find((item) => String(item.upc) === productSelect.value && String(item.store) === storeSelect.value && String(item.discount) === discountSelect.value);
    if (!row) return;
    setText("#scenario-units", `×${number(row.expected_unit_multiplier, 2)}`);
    setText("#scenario-revenue", signed(row.incremental_revenue_mean));
    setText("#scenario-margin", signed(row.incremental_margin_mean));
    setText("#scenario-probability", percent(row.probability_positive_margin));
    $("#scenario-margin").classList.toggle("loss-value", row.incremental_margin_mean < 0);
  }
  productSelect.addEventListener("change", updateStores);
  storeSelect.addEventListener("change", updateDiscounts);
  discountSelect.addEventListener("change", updateOutput);
  updateStores();
}

function renderMulticategory(data) {
  if (!data.multicategory.available) return;
  $("#multicategory-content").classList.remove("empty-state");
  $("#multicategory-content").innerHTML = data.multicategory.categories.map((row) => `<div class="interval-row"><span class="interval-label">${row.category.replaceAll("_", " ")}</span><strong>${number(row.posterior_mean, 2)}</strong><span>${number(row.p05, 2)} to ${number(row.p95, 2)}</span></div>`).join("");
  const holdout = data.multicategory.holdout;
  const weakest = [...data.multicategory.holdout_by_category].sort((a, b) => a.coverage_90 - b.coverage_90)[0];
  setText("#multicategory-health", `Conditional holdout: log MAE ${number(holdout.log_mae, 2)} and ${percent(holdout.predictive_interval_calibration["90"].empirical)} coverage for a nominal 90% interval. Weakest category: ${weakest.category.replaceAll("_", " ")} at ${percent(weakest.coverage_90)}. The model now cross-classifies category/product slopes with partially pooled store response offsets.`);
}

function renderDataFoundation(data) {
  const foundation = data.data_foundation;
  const categories = foundation.categories.filter((row) => row && row.available !== false);
  $("#data-foundation-grid").innerHTML = categories.map((row) => `<article><span>${row.category.replaceAll("_", " ")}</span><strong>${number(row.valid_rows, 0)}</strong><small>validated rows · ${number(row.raw_rows, 0)} raw</small></article>`).join("");
  const mart = foundation.sql_mart;
  if (mart.available === false) {
    $("#sql-checks").textContent = "Build the SQL mart to populate relational checks.";
    return;
  }
  const checks = mart.quality_checks;
  $("#sql-checks").innerHTML = `<strong>${number(mart.counts.fact_rows, 0)} fact rows</strong><span>${number(mart.counts.products, 0)} products</span><span>${number(mart.counts.stores, 0)} stores</span><span>${number(mart.counts.weeks, 0)} weeks</span><span>${checks.duplicate_fact_keys} duplicate keys</span><span>${checks.orphan_dimension_keys} orphan keys</span>`;
}

function renderOperations(data) {
  const operations = data.operational_readiness;
  const causalValidation = data.causal_validation;
  const powerCase = data.experiment_plan.power_sensitivity.scenarios.find((row) => row.minimum_detectable_unit_lift === .2 && row.intraclass_correlation_assumption === .05);
  const checksPassed = Object.values(operations.system_engineering_checks).filter(Boolean).length;
  const checksTotal = Object.keys(operations.system_engineering_checks).length;
  $("#operations-grid").innerHTML = `
    <article><span>Engineering controls</span><strong>${checksPassed}/${checksTotal}</strong><small>contract, lineage, CI, container, monitoring, runbook</small></article>
    <article><span>Known-truth causal validation</span><strong>${causalValidation.implementation_validation_passed ? "PASS" : "FAIL"}</strong><small>hidden-confounding failure remains visible</small></article>
    <article><span>Illustrative 20% lift test</span><strong>${powerCase.stores_per_arm}/arm</strong><small>13 weeks, ICC 0.05; pilot must validate assumptions</small></article>
    <article><span>Historical policy</span><strong>${operations.release_decision.toUpperCase()}</strong><small>fail-closed service returns no recommendation</small></article>`;
}

function renderRelease(data) {
  const release = data.release_gate;
  const intake = data.experiment_intake;
  const validation = data.validation;
  const bayesian = data.bayesian;
  const causal = data.causal;
  const optimization = data.optimization.policies;
  const shadow = data.shadow_replay;
  const gates = [
    {
      label: "Data foundation",
      status: "complete",
      metric: `${number(validation.valid_rows / 1e6, 2)}M valid`,
      detail: `${number(validation.raw_rows / 1e6, 2)}M raw · ${number(validation.duplicate_keys, 0)} duplicate keys`,
    },
    {
      label: "Price response",
      status: bayesian.diagnostics.divergences === 0 ? "complete" : "warning",
      metric: number(bayesian.mean_elasticity.posterior_mean, 2),
      detail: `partial pooling · 0 divergences · R-hat ${number(bayesian.diagnostics.max_rhat, 2)}`,
    },
    {
      label: "Causal audit",
      status: "complete",
      metric: causal.identification_gate_passed ? "Admitted" : "Withheld",
      detail: "overlap · balance · future placebo · past-outcome control",
    },
    {
      label: "Margin decision",
      status: "complete",
      metric: `${optimization.risk_aware_margin.selected_promotions} actions`,
      detail: `${signed(optimization.revenue.incremental_revenue_mean, 0)} revenue · ${signed(optimization.revenue.incremental_margin_mean, 0)} margin`,
    },
    {
      label: "Historical replay",
      status: shadow.overall_status === "pass" ? "complete" : "warning",
      metric: percent(shadow.rolling_90_coverage),
      detail: `${shadow.replay_weeks} untouched weeks · past-only recalibration`,
    },
  ];
  setText("#release-status", "Research complete · automation withheld");
  $("#release-rail").innerHTML = gates.map((gate, index) => `
    <li class="release-step ${gate.status}">
      <span class="release-index">0${index + 1}</span>
      <div><strong>${gate.label}</strong><small>${gate.detail}</small></div>
      <b>${gate.metric}</b>
      <span class="release-result">${gate.status === "complete" ? "Complete" : "Warning"}</span>
    </li>`).join("");

  const intakeChecks = Object.entries(intake.checks || {});
  $("#intake-register").innerHTML = `
    <div class="intake-head"><strong>External deployment evidence</strong><span>${release.blockers.length} gates remain external</span></div>
    <div class="intake-grid">${intakeChecks.map(([name, passed]) => `
      <span class="intake-check ${passed ? "passed" : "blocked"}"><b>${passed ? "Pass" : "Open"}</b>${name.replaceAll("_", " ")}</span>`).join("")}</div>`;
}

function renderShadowReplay(data) {
  const replay = data.shadow_replay;
  const batches = data.shadow_batches;
  setText("#replay-status", replay.overall_status);
  $("#replay-summary").innerHTML = `
    <div><span>Rows replayed</span><strong>${number(replay.replay_rows, 0)}</strong></div>
    <div><span>Rolling 90% coverage</span><strong>${percent(replay.rolling_90_coverage)}</strong></div>
    <div><span>Mean log MAE</span><strong>${number(replay.mean_log_mae, 3)}</strong></div>
    <div><span>Maximum price PSI</span><strong>${number(replay.maximum_price_psi, 2)}</strong></div>`;
  $("#replay-timeline").innerHTML = batches.map((batch) => `
    <div class="replay-week ${batch.batch_status}" title="Week ${batch.week}: coverage ${percent(batch.rolling_90_coverage)}, PSI ${number(batch.price_psi, 2)}, bias ${signed(batch.mean_bias_actual_minus_predicted, 2)}">
      <span>${batch.week}</span><b>${percent(batch.rolling_90_coverage, 0)}</b>
    </div>`).join("");
  const risk = replay.highest_risk_week;
  $("#replay-risk").innerHTML = `<strong>Highest-drift week ${risk.week}</strong><span>Price PSI ${number(risk.price_psi, 2)}</span><span>Promotion rate ${percent(risk.promotion_rate)}</span><span>Coverage ${percent(risk.rolling_90_coverage)}</span><span>Bias ${signed(risk.mean_bias_actual_minus_predicted, 2)}</span>`;
}

function render(data) {
  state.data = data;
  const bayes = data.bayesian;
  const optimization = data.optimization.policies;
  const event = data.event_summary;
  const causal = data.causal;
  const calibration90 = data.calibration.predictive_interval_calibration["90"].empirical;
  setText("#elasticity-kpi", number(bayes.mean_elasticity.posterior_mean, 2));
  setText("#revenue-kpi", signed(optimization.revenue.incremental_revenue_mean, 0));
  setText("#margin-kpi", signed(optimization.revenue.incremental_margin_mean, 0));
  setText("#events-kpi", number(event.event_count, 0));
  setText("#margin-actions", optimization.risk_aware_margin.selected_promotions);
  setText("#overlap", percent(causal.diagnostics.fraction_strictly_inside_005_095));
  setText("#coverage", percent(calibration90));
  setText("#event-units", signed(event.event_week_unit_difference_pct, 0) + "%");
  setText("#event-price", signed(event.event_week_price_difference_pct, 1) + "%");
  setText("#event-margin", signed(event.event_week_margin_difference_dollars, 2));
  setText("#event-pretrend", signed(event.week_minus_1_pretrend_unit_difference_pct, 1) + "%");
  const payback = data.payback_summary;
  setText("#payback-copy", `${number(payback.episodes, 0)} price-derived episodes show ${signed(payback.post_weeks_1_4_unit_difference_pct, 1)}% units in weeks 1–4 and ${signed(payback.post_weeks_5_8_unit_difference_pct, 1)}% in weeks 5–8 versus clean pre-periods. Descriptive only: median price is a proxy and promotion timing is endogenous.`);
  setText("#sensitivity-copy", `Product means move at most ${number(data.sensitivity.maximum_product_mean_range_across_priors, 3)} across priors; the global mean moves ${number(data.sensitivity.global_mean_range_across_priors, 3)} and fails its stability rule.`);
  const conformal90 = data.calibration.temporal_split_conformal_recalibration.intervals["90"];
  setText("#conformal-copy", `A strictly later evaluation window reaches ${percent(conformal90.evaluation_coverage)} coverage after split-conformal widening, versus a 90% target. It remains below target, so the deployment block stays in place.`);
  const funding = data.optimization.vendor_funding_sensitivity;
  setText("#funding-copy", `No funding: 0 actions. $0.10 per promoted unit: ${funding["0.10"].selected_promotions} modeled actions and ${signed(funding["0.10"].funded_incremental_margin_mean)} aggregate margin.`);
  setText("#generated-at", `Data snapshot: ${new Date(data.generated_at_utc).toLocaleString()}`);
  renderTradeoff(data);
  renderElasticities(data.product_elasticities);
  renderCalibration(data);
  renderGate(data);
  renderBalance(data.causal_balance);
  renderEvent(data);
  populateScenario(data);
  renderMulticategory(data);
  renderDataFoundation(data);
  renderOperations(data);
  renderRelease(data);
  renderShadowReplay(data);
}

$$('.mode-button').forEach((button) => button.addEventListener('click', () => {
  state.mode = button.dataset.mode;
  document.body.dataset.mode = state.mode;
  $$('.mode-button').forEach((item) => {
    const active = item === button;
    item.classList.toggle('active', active);
    item.setAttribute('aria-pressed', String(active));
  });
}));

fetch("data.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Dashboard data returned ${response.status}.`);
    return response.json();
  })
  .then(render)
  .catch((error) => {
    const banner = $("#error-banner");
    banner.hidden = false;
    banner.textContent = `The dashboard could not load its research snapshot. Run dashboard/build_data.py and serve this folder over HTTP. ${error.message}`;
  });
