"""Untrusted input validation for predictions.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("sample_id", "split", "y_true", "y_pred", "confidence")
MAX_ROWS = 200_000


class DataValidationError(ValueError):
    """Raised when prediction data cannot be trusted for analysis."""


def load_predictions(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser().resolve(strict=True)
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise DataValidationError("Input must be a readable .csv file")
    if csv_path.stat().st_size > 35_000_000:
        raise DataValidationError("CSV exceeds 35 MB limit")
    try:
        df = pd.read_csv(csv_path, dtype={"sample_id": "string", "split": "string", "y_true": "string", "y_pred": "string"})
    except Exception as exc:
        raise DataValidationError(f"Cannot load CSV: {exc}") from exc

    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if not 1 <= len(df) <= MAX_ROWS:
        raise DataValidationError(f"Dataset must contain 1..{MAX_ROWS} rows")

    for col in ("sample_id", "split", "y_true", "y_pred"):
        df[col] = df[col].astype("string").str.strip()
        if df[col].isna().any() or df[col].eq("").any():
            raise DataValidationError(f"Missing or empty values in '{col}'")

    df["split"] = df["split"].str.lower()
    if not set(df["split"]).issubset({"train", "val", "test"}):
        raise DataValidationError("split values must be: train, val, test")

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    confidence = df["confidence"].to_numpy(dtype=float)
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise DataValidationError("confidence must be numeric, finite and in [0,1]")

    if "group" in df.columns:
        df["group"] = df["group"].astype("string").fillna("unknown").str.strip().replace("", "unknown")
    if "image_path" in df.columns:
        df["image_path"] = df["image_path"].astype("string").fillna("").str.strip()

    return df
