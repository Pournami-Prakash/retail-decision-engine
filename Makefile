.PHONY: test lint verify dashboard-data operational release-evidence shadow-replay

test:
	.venv/bin/python -m unittest discover -s tests -v

lint:
	.venv/bin/ruff check src tests dashboard

dashboard-data:
	.venv/bin/python dashboard/build_data.py

operational:
	.venv/bin/retail-decision operational-check

release-evidence:
	.venv/bin/retail-decision experiment-analyze --category cereal --input templates/promotion_experiment_observations.csv
	.venv/bin/retail-decision release-gate --category cereal

shadow-replay:
	.venv/bin/retail-decision shadow-replay --category cereal

verify: lint test shadow-replay release-evidence operational dashboard-data
	.venv/bin/python -m compileall -q src dashboard
	.venv/bin/python -m json.tool contracts/decision_request.schema.json >/dev/null
	.venv/bin/python -m json.tool contracts/decision_response.schema.json >/dev/null
	.venv/bin/python -m json.tool contracts/store_item_week.schema.json >/dev/null
	.venv/bin/python -m json.tool contracts/promotion_experiment_observation.schema.json >/dev/null
