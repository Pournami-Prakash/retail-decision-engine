from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR, RAW_DIR, ensure_directories


MOVEMENT_REQUIRED = {"upc", "store", "week", "move", "qty", "price", "profit", "sale", "ok"}
UPC_REQUIRED = {"upc", "com_code", "nitem", "descrip", "size", "case"}


@dataclass
class ValidationReport:
    category: str
    raw_rows: int
    valid_rows: int
    invalid_source_rows: int
    duplicate_keys: int
    missing_key_rows: int
    nonpositive_price_rows: int
    nonpositive_quantity_rows: int
    negative_movement_rows: int
    unmatched_upc_rows: int
    sale_flag_rate: float


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _read_movement(category: str) -> pd.DataFrame:
    archive = RAW_DIR / category / "movement.zip"
    if not archive.exists():
        raise FileNotFoundError(f"Missing {archive}; run the download command first")
    with zipfile.ZipFile(archive) as zipped:
        csv_members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected one CSV in {archive}, found {csv_members}")
        with zipped.open(csv_members[0]) as handle:
            return _normalize_columns(pd.read_csv(handle, low_memory=False, encoding="cp1252"))


def _read_upc(category: str) -> pd.DataFrame:
    path = RAW_DIR / category / "upc.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the download command first")
    # The historical product descriptions include Windows-1252 bytes.
    return _normalize_columns(pd.read_csv(path, low_memory=False, encoding="cp1252"))


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def build_category_panel(category: str) -> tuple[Path, Path]:
    ensure_directories()
    movement = _read_movement(category)
    upc = _read_upc(category)
    _require_columns(movement, MOVEMENT_REQUIRED, "movement")
    _require_columns(upc, UPC_REQUIRED, "UPC")

    numeric_movement = ["upc", "store", "week", "move", "qty", "price", "profit", "ok"]
    for column in numeric_movement:
        movement[column] = pd.to_numeric(movement[column], errors="coerce")
    upc["upc"] = pd.to_numeric(upc["upc"], errors="coerce")

    raw_rows = len(movement)
    duplicate_keys = int(movement.duplicated(["upc", "store", "week"], keep=False).sum())
    missing_key_rows = int(movement[["upc", "store", "week"]].isna().any(axis=1).sum())
    invalid_source_rows = int((movement["ok"] != 1).sum())
    nonpositive_price_rows = int((movement["price"] <= 0).sum())
    nonpositive_quantity_rows = int((movement["qty"] <= 0).sum())
    negative_movement_rows = int((movement["move"] < 0).sum())

    clean = movement.loc[
        (movement["ok"] == 1)
        & movement[["upc", "store", "week", "move", "qty", "price"]].notna().all(axis=1)
        & (movement["price"] > 0)
        & (movement["qty"] > 0)
        & (movement["move"] >= 0)
    ].copy()
    clean = clean.drop_duplicates(["upc", "store", "week"], keep="last")

    # The source manual specifies revenue = price * move / qty for bundle offers.
    clean["unit_price"] = clean["price"] / clean["qty"]
    clean["revenue"] = clean["unit_price"] * clean["move"]
    clean["gross_margin_rate"] = clean["profit"] / 100.0
    clean["gross_margin_dollars"] = clean["revenue"] * clean["gross_margin_rate"]
    clean["promotion_observed"] = clean["sale"].fillna("").astype(str).str.strip().ne("")
    clean["log_units"] = np.log1p(clean["move"])
    clean["log_unit_price"] = np.log(clean["unit_price"])
    clean["category"] = category

    product_columns = ["upc", "com_code", "nitem", "descrip", "size", "case"]
    products = upc[product_columns].drop_duplicates("upc", keep="last")
    panel = clean.merge(products, on="upc", how="left", validate="many_to_one")
    unmatched_upc_rows = int(panel["descrip"].isna().sum())

    report = ValidationReport(
        category=category,
        raw_rows=raw_rows,
        valid_rows=len(panel),
        invalid_source_rows=invalid_source_rows,
        duplicate_keys=duplicate_keys,
        missing_key_rows=missing_key_rows,
        nonpositive_price_rows=nonpositive_price_rows,
        nonpositive_quantity_rows=nonpositive_quantity_rows,
        negative_movement_rows=negative_movement_rows,
        unmatched_upc_rows=unmatched_upc_rows,
        sale_flag_rate=float(panel["promotion_observed"].mean()),
    )

    output_path = PROCESSED_DIR / f"{category}_store_product_week.csv.gz"
    report_path = ARTIFACT_DIR / f"{category}_validation.json"
    panel.to_csv(output_path, index=False, compression="gzip")
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return output_path, report_path
