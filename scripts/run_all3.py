# scripts/run_all3.py
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_all2 import main as run_all2_main


CYCLICAL_PERIODS: dict[str, float] = {
    "hour": 24.0,
    "dayofweek": 7.0,
    "month": 12.0,
    "dayofyear": 365.25,
}

# These raw time columns are replaced rather than retained alongside their
# cyclic encodings. time_idx remains the TimeSeriesDataSet index, but is not
# used as a real-valued covariate. year is also removed to avoid fitting a
# spurious linear trend across the four-year Utrecht series.
RAW_TIME_REALS = {
    "time_idx",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "year",
}


def _cyclical_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column == "hour":
        values = df[column].to_numpy(dtype=float)
    elif column in {"dayofweek", "dayofyear"}:
        values = df[column].to_numpy(dtype=float)
        if column == "dayofyear":
            values = values - 1.0
    elif column == "month":
        values = df[column].to_numpy(dtype=float) - 1.0
    else:
        raise ValueError(f"Unsupported cyclical time column: {column}")
    return values


def add_cyclical_time_features(
    df_full: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Replace raw calendar reals with deterministic sine/cosine features."""
    if "timestamp" not in df_full.columns:
        raise ValueError(
            "A timestamp column is required to generate cyclical time features."
        )

    timestamp = pd.to_datetime(df_full["timestamp"], errors="coerce")
    if timestamp.isna().any():
        invalid_count = int(timestamp.isna().sum())
        raise ValueError(
            f"timestamp contains {invalid_count} invalid values, so calendar "
            "features cannot be generated."
        )

    calendar_values = {
        "hour": timestamp.dt.hour,
        "dayofweek": timestamp.dt.dayofweek,
        "month": timestamp.dt.month,
        "dayofyear": timestamp.dt.dayofyear,
        "year": timestamp.dt.year,
    }
    derived_columns: list[str] = []
    for column, values in calendar_values.items():
        if column not in df_full.columns:
            df_full[column] = values
            derived_columns.append(column)

    missing = [column for column in CYCLICAL_PERIODS if column not in df_full.columns]
    if missing:
        raise ValueError(f"Missing columns required for cyclical encoding: {missing}")

    generated: list[str] = []
    for column, period in CYCLICAL_PERIODS.items():
        values = _cyclical_values(df_full, column)
        angle = 2.0 * np.pi * values / period
        sin_col = f"{column}_sin"
        cos_col = f"{column}_cos"
        df_full[sin_col] = np.sin(angle).astype(np.float32)
        df_full[cos_col] = np.cos(angle).astype(np.float32)
        generated.extend([sin_col, cos_col])

    features = cfg["features"]
    requested_known = list(features.get("known_reals", []))
    retained_known = [column for column in requested_known if column not in RAW_TIME_REALS]
    features["known_reals"] = list(dict.fromkeys(retained_known + generated))

    # Generated calendar features are deterministically available throughout
    # the forecast horizon, so permit them in the decoder.
    future_available = list(features.get("future_available_reals", []))
    features["future_available_reals"] = list(
        dict.fromkeys(future_available + generated)
    )

    return {
        "name": "cyclical_time_features",
        "generated_reals": generated,
        "derived_calendar_columns": derived_columns,
        "removed_known_reals": sorted(RAW_TIME_REALS.intersection(requested_known)),
        "periods": CYCLICAL_PERIODS,
    }


def main(config_path: str) -> None:
    run_all2_main(
        config_path,
        dataframe_transform=add_cyclical_time_features,
        run_suffix="_all3",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
