from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .config import ARTIFACT_DIR, PROCESSED_DIR


ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "sql" / "retail_mart.sql"


def build_sql_mart(category: str) -> tuple[Path, Path]:
    panel_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {panel_path}; build the analytical panel first")
    database_path = PROCESSED_DIR / f"{category}_retail_mart.duckdb"
    escaped_path = str(panel_path.resolve()).replace("'", "''")
    sql = SQL_PATH.read_text(encoding="utf-8").replace("{{panel_path}}", escaped_path)

    with duckdb.connect(str(database_path)) as connection:
        connection.execute(sql)
        metrics = connection.execute(
            """
            SELECT
                count(*) AS fact_rows,
                count(DISTINCT (store_key, product_key, week_key)) AS distinct_grain,
                count_if(units <= 0) AS nonpositive_units,
                count_if(unit_price <= 0) AS nonpositive_prices,
                count_if(NOT isfinite(revenue)) AS nonfinite_revenue,
                count_if(NOT isfinite(gross_margin_dollars)) AS nonfinite_margin
            FROM fact_store_product_week
            """
        ).fetchone()
        dimensions = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM dim_product) AS products,
                (SELECT count(*) FROM dim_store) AS stores,
                (SELECT count(*) FROM dim_week) AS weeks
            """
        ).fetchone()
        orphan_count = connection.execute(
            """
            SELECT count(*)
            FROM fact_store_product_week f
            LEFT JOIN dim_store s USING (store_key)
            LEFT JOIN dim_product p USING (product_key)
            LEFT JOIN dim_week w USING (week_key)
            WHERE s.store_key IS NULL OR p.product_key IS NULL OR w.week_key IS NULL
            """
        ).fetchone()[0]

    report = {
        "category": category,
        "database": str(database_path),
        "schema": {
            "fact": "fact_store_product_week",
            "dimensions": ["dim_product", "dim_store", "dim_week"],
            "grain": "one row per store, product, and retail week",
        },
        "counts": {
            "fact_rows": metrics[0],
            "distinct_fact_grain": metrics[1],
            "products": dimensions[0],
            "stores": dimensions[1],
            "weeks": dimensions[2],
        },
        "quality_checks": {
            "duplicate_fact_keys": metrics[0] - metrics[1],
            "orphan_dimension_keys": orphan_count,
            "nonpositive_units": metrics[2],
            "nonpositive_prices": metrics[3],
            "nonfinite_revenue": metrics[4],
            "nonfinite_margin": metrics[5],
        },
    }
    report_path = ARTIFACT_DIR / f"{category}_sql_mart_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return database_path, report_path
