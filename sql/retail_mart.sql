CREATE OR REPLACE TEMP VIEW source_panel AS
SELECT *
FROM read_csv_auto('{{panel_path}}', header = true, sample_size = -1);

CREATE OR REPLACE TABLE dim_product AS
SELECT
    row_number() OVER (ORDER BY upc)::INTEGER AS product_key,
    upc::BIGINT AS upc,
    any_value(category)::VARCHAR AS category,
    any_value(descrip)::VARCHAR AS description,
    any_value(size)::VARCHAR AS package_size,
    any_value("case")::INTEGER AS case_pack
FROM source_panel
GROUP BY upc;

CREATE OR REPLACE TABLE dim_store AS
SELECT
    row_number() OVER (ORDER BY store)::INTEGER AS store_key,
    store::INTEGER AS source_store_id
FROM source_panel
GROUP BY store;

CREATE OR REPLACE TABLE dim_week AS
SELECT
    week::INTEGER AS week_key,
    ((week - 1) % 52 + 1)::INTEGER AS retail_week_of_year,
    floor((week - 1) / 52)::INTEGER AS relative_year,
    sin(2 * pi() * week / 52.0) AS season_sin,
    cos(2 * pi() * week / 52.0) AS season_cos
FROM source_panel
GROUP BY week;

CREATE OR REPLACE TABLE fact_store_product_week AS
SELECT
    s.store_key,
    p.product_key,
    x.week::INTEGER AS week_key,
    x.move::DOUBLE AS units,
    x.unit_price::DOUBLE AS unit_price,
    x.revenue::DOUBLE AS revenue,
    x.gross_margin_rate::DOUBLE AS gross_margin_rate,
    x.gross_margin_dollars::DOUBLE AS gross_margin_dollars,
    coalesce(cast(x.sale AS VARCHAR), '') AS promotion_code,
    x.promotion_observed::BOOLEAN AS promotion_observed,
    x.log_units::DOUBLE AS log_units,
    x.log_unit_price::DOUBLE AS log_unit_price
FROM source_panel x
JOIN dim_store s ON x.store = s.source_store_id
JOIN dim_product p ON x.upc = p.upc;

CREATE UNIQUE INDEX IF NOT EXISTS idx_fact_grain
ON fact_store_product_week (store_key, product_key, week_key);

CREATE INDEX IF NOT EXISTS idx_fact_week
ON fact_store_product_week (week_key);

