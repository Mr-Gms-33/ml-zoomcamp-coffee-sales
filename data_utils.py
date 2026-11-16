"""Load coffee sales data and build model features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"

NUMERIC_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]
CATEGORICAL_FEATURES = ["cash_type", "has_card"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "coffee_name"


def load_machine_one(path: Path | None = None) -> pd.DataFrame:
    path = path or ARCHIVE_DIR / "index_1.csv"
    df = pd.read_csv(path)
    df["machine_id"] = "machine_1"
    return df


def load_machine_two(path: Path | None = None) -> pd.DataFrame:
    path = path or ARCHIVE_DIR / "index_2.csv"
    df = pd.read_csv(path)
    df["machine_id"] = "machine_2"
    df["card"] = pd.NA
    return df


def load_all_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_machine_one(), load_machine_two()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    card = out["card"]
    out["has_card"] = (
        card.notna() & card.astype(str).str.strip().ne("") & card.astype(str).ne("nan")
    ).map({True: "yes", False: "no"})
    out["cash_type"] = out["cash_type"].astype(str).str.strip().str.lower()
    return out


def get_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Primary model: machine 1 (consistent drink catalog)."""
    df = engineer_features(load_machine_one())
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return x, y