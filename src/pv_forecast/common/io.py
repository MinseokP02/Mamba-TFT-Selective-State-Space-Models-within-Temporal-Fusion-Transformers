# src/pv_forecast/utils/io.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_dataframe(csv_path: str, parse_dates: List[str] | None = None, index_col: str | None = None) -> pd.DataFrame:
    if parse_dates is None:
        parse_dates = []
    return pd.read_csv(csv_path, parse_dates=parse_dates, index_col=index_col)




def predictions_to_dataframe(pred_outputs) -> pd.DataFrame:
    rows = []

    for batch_out in pred_outputs:
        y_hat = batch_out["y_hat"].numpy()
        y = batch_out["y"].numpy()
        meta = batch_out["meta"]

        batch_size, horizon = y_hat.shape

        group_ids = meta["group_id"]
        dec_start = meta["decoder_time_idx_start"]
        dec_end = meta["decoder_time_idx_end"]

        for i in range(batch_size):
            row = {
                "group_id": group_ids[i],
                "decoder_time_idx_start": int(dec_start[i]),
                "decoder_time_idx_end": int(dec_end[i]),
            }

            for h in range(horizon):
                row[f"y_true_t{h+1}"] = float(y[i, h])
                row[f"y_pred_t{h+1}"] = float(y_hat[i, h])

            rows.append(row)

    return pd.DataFrame(rows)

def predictions_to_dataframe(
    pred_outputs,
    df_full: pd.DataFrame,
    quantiles=None,
    time_idx_col: str = "time_idx",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    # time_idx -> timestamp 매핑
    time_map = (
        df_full[[time_idx_col, timestamp_col]]
        .drop_duplicates(subset=[time_idx_col])
        .sort_values(time_idx_col)
        .set_index(time_idx_col)[timestamp_col]
    )

    rows = []

    for batch_out in pred_outputs:
        y_hat = batch_out["y_hat"].numpy()   # (B,H,Q) or (B,H)
        y = batch_out["y"].numpy()           # (B,H)
        meta = batch_out["meta"]

        group_ids = meta["group_id"]
        dec_start = meta["decoder_time_idx_start"]
        dec_end = meta["decoder_time_idx_end"]

        # decoder 각 step의 실제 time_idx 만들기
        if y_hat.ndim == 2:
            batch_size, horizon = y_hat.shape
            num_quantiles = None
        elif y_hat.ndim == 3:
            batch_size, horizon, num_quantiles = y_hat.shape
            if num_quantiles != len(quantiles):
                raise ValueError(
                    f"y_hat num_quantiles={num_quantiles} != len(quantiles)={len(quantiles)}"
                )
        else:
            raise ValueError(f"Unsupported y_hat shape: {y_hat.shape}")

        for i in range(batch_size):
            start_idx = int(dec_start[i])

            for h in range(horizon):
                decoder_time_idx = start_idx + h
                ts = time_map.get(decoder_time_idx, pd.NaT)

                row = {
                    "group_id": group_ids[i],
                    "decoder_time_idx_start": start_idx,
                    "decoder_time_idx_end": int(dec_end[i]),
                    "horizon_idx": h + 1,
                    "decoder_time_idx": decoder_time_idx,
                    "timestamp": pd.to_datetime(ts),
                    "y_true": float(y[i, h]),
                }

                if y_hat.ndim == 2:
                    row["y_pred"] = float(y_hat[i, h])
                else:
                    for q_idx, q in enumerate(quantiles):
                        q_name = int(q * 100)
                        row[f"y_pred_q{q_name}"] = float(y_hat[i, h, q_idx])

                rows.append(row)

    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out = out.sort_values(["group_id", "timestamp", "horizon_idx"]).reset_index(drop=True)
    return out


def postprocess_cnn_predictions(pred_outputs, df_full: pd.DataFrame, quantiles=None):
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    time_map = (
        df_full[["time_idx", "timestamp"]]
        .drop_duplicates()
        .set_index("time_idx")["timestamp"]
    )

    rows = []
    sample_offset = 0

    for batch_out in pred_outputs:
        y_hat = batch_out["y_hat"].numpy()   # (B,H,Q)
        y = batch_out["y"].numpy()           # (B,H)
        meta = batch_out["meta"]

        dec_start = np.asarray(meta["decoder_time_idx_start"]).astype(np.int64)
        batch_size, horizon, num_quantiles = y_hat.shape

        if num_quantiles != len(quantiles):
            raise ValueError(
                f"num_quantiles={num_quantiles}, len(quantiles)={len(quantiles)}"
            )

        for i in range(batch_size):
            sample_id = sample_offset + i
            decoder_time_idx = dec_start[i] + np.arange(horizon, dtype=np.int64)
            ts = pd.to_datetime(time_map.loc[decoder_time_idx].to_numpy())
            forecast_start = ts[0]

            for h in range(horizon):
                row = {
                    "sample": sample_id,
                    "forecast_start": forecast_start,
                    "horizon_idx": h + 1,
                    "timestamp": ts[h],
                    "date": pd.Timestamp(ts[h]).floor("D"),
                    "hour": pd.Timestamp(ts[h]).hour,
                    "true": float(y[i, h]),
                    "pred_q10": float(y_hat[i, h, 0]),
                    "pred_q50": float(y_hat[i, h, 1]),
                    "pred_q90": float(y_hat[i, h, 2]),
                }
                rows.append(row)

        sample_offset += batch_size

    out_all = pd.DataFrame(rows)

    out_all["err"] = out_all["pred_q50"] - out_all["true"]
    out_all["abs_err"] = out_all["err"].abs()
    out_all["sq_err"] = out_all["err"] ** 2

    final_df = out_all.sort_values(["date", "timestamp", "sample"]).reset_index(drop=True)

    one = (
        final_df
        .groupby("timestamp", as_index=False)
        .agg(
            true=("true", "first"),
            pred_q10=("pred_q10", "mean"),
            pred_q50=("pred_q50", "mean"),
            pred_q90=("pred_q90", "mean"),
        )
    )
    one["date"] = one["timestamp"].dt.floor("D")
    one["hour"] = one["timestamp"].dt.hour
    one["err"] = one["pred_q50"] - one["true"]
    one["abs_err"] = one["err"].abs()
    one["sq_err"] = one["err"] ** 2
    one["in_80"] = ((one["true"] >= one["pred_q10"]) & (one["true"] <= one["pred_q90"])).astype(int)

    return final_df, one
