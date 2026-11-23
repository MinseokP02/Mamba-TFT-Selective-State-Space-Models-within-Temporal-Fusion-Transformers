# src/pv_forecast/scripts_like/eval_cnn_impl.py

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from pv_forecast.common.io import load_yaml, ensure_dir
from pv_forecast.common.paths import save_json


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    mean pinball loss
    q=0.5일 때는 MAE의 절반 값이 나옴.
    """
    err = y_true - y_pred
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.mean(loss))


def _quantile_risk(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    TFT 논문식 q-Risk 형태:
    q-Risk = 2 * sum(QuantileLoss) / sum(abs(y_true))
    """
    err = y_true - y_pred
    loss = np.maximum(q * err, (q - 1.0) * err)

    denom = np.sum(np.abs(y_true))
    if denom <= 0:
        return float("nan")

    return float(2.0 * np.sum(loss) / denom)


def _load_split_info(run_dir: Path, cfg: dict) -> dict:
    """
    train_meta.json이 있으면 거기에서 split 정보를 가져오고,
    없으면 config의 split ratio만 가져옴.
    """
    meta_path = run_dir / "meta" / "train_meta.json"

    split_cfg = cfg.get("split", {})
    default_split = {
        "train_ratio": float(split_cfg.get("train_ratio", 0.7)),
        "val_ratio": float(split_cfg.get("val_ratio", 0.15)),
        "test_ratio": float(split_cfg.get("test_ratio", 0.15)),
    }

    if not meta_path.exists():
        return default_split

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "train_ratio": meta.get("train_ratio", default_split["train_ratio"]),
        "val_ratio": meta.get("val_ratio", default_split["val_ratio"]),
        "test_ratio": meta.get("test_ratio", default_split["test_ratio"]),

        "train_cutoff": meta.get("train_cutoff", meta.get("cutoff", None)),
        "val_cutoff": meta.get("val_cutoff", meta.get("validation_cutoff", None)),

        "train_cutoff_time_idx": meta.get(
            "train_cutoff_time_idx",
            meta.get("cutoff_time_idx", meta.get("train_cutoff", None)),
        ),
        "val_cutoff_time_idx": meta.get(
            "val_cutoff_time_idx",
            meta.get("validation_cutoff", meta.get("val_cutoff", None)),
        ),
    }


def eval_cnn_main(config_path: str, run_dir: str):
    cfg = load_yaml(config_path)
    run_dir = Path(run_dir)

    pred_path = run_dir / "tables" / "pred_one.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path)

    required_cols = ["timestamp", "true", "pred_q50"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"pred_one.csv must contain '{col}' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if len(df) == 0:
        raise ValueError("No prediction rows found in eval_cnn_main")

    y_true = df["true"].to_numpy(dtype=float)
    y_pred_q50 = df["pred_q50"].to_numpy(dtype=float)

    mae_p50 = _mae(y_true, y_pred_q50)
    rmse_p50 = _rmse(y_true, y_pred_q50)

    metrics = {
        "n": int(len(df)),
        "mae_p50": mae_p50,
        "rmse_p50": rmse_p50,
    }

    has_q10 = "pred_q10" in df.columns
    has_q90 = "pred_q90" in df.columns

    if has_q10 and has_q90:
        y_pred_q10 = df["pred_q10"].to_numpy(dtype=float)
        y_pred_q90 = df["pred_q90"].to_numpy(dtype=float)

        coverage_10_90 = float(
            np.mean((y_true >= y_pred_q10) & (y_true <= y_pred_q90))
        )

        width_10_90 = float(np.mean(y_pred_q90 - y_pred_q10))

        qloss_q10 = _quantile_loss(y_true, y_pred_q10, 0.1)
        qloss_q50 = _quantile_loss(y_true, y_pred_q50, 0.5)
        qloss_q90 = _quantile_loss(y_true, y_pred_q90, 0.9)

        qrisk_q10 = _quantile_risk(y_true, y_pred_q10, 0.1)
        qrisk_q50 = _quantile_risk(y_true, y_pred_q50, 0.5)
        qrisk_q90 = _quantile_risk(y_true, y_pred_q90, 0.9)

        metrics.update(
            {
                "coverage_10_90": coverage_10_90,
                "width_10_90": width_10_90,
                "qloss_q10": qloss_q10,
                "qloss_q50": qloss_q50,
                "qloss_q90": qloss_q90,
                "qrisk_q10": qrisk_q10,
                "qrisk_q50": qrisk_q50,
                "qrisk_q90": qrisk_q90,
            }
        )
    else:
        # q10/q90이 없더라도 q50 quantile loss는 계산 가능
        metrics["qloss_q50"] = _quantile_loss(y_true, y_pred_q50, 0.5)
        metrics["qrisk_q50"] = _quantile_risk(y_true, y_pred_q50, 0.5)

    # split 정보 추가
    metrics["split"] = _load_split_info(run_dir, cfg)

    report_dir = run_dir / "reports"
    ensure_dir(report_dir)

    # 원하는 형태의 JSON 저장
    metrics_path = report_dir / "metrics.json"
    save_json(metrics, metrics_path)

    # 기존처럼 CSV도 같이 저장하고 싶으면 유지
    flat_metrics = {k: v for k, v in metrics.items() if k != "split"}
    report_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in flat_metrics.items()]
    )
    report_csv_path = report_dir / "metrics.csv"
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    # 시간대별 coverage 저장
    hourly_path = None
    if has_q10 and has_q90:
        tmp = df.copy()
        tmp["hour"] = tmp["timestamp"].dt.hour
        tmp["in_10_90"] = (
            (tmp["true"] >= tmp["pred_q10"]) &
            (tmp["true"] <= tmp["pred_q90"])
        ).astype(int)
        tmp["width_10_90"] = tmp["pred_q90"] - tmp["pred_q10"]
        tmp["abs_error_p50"] = np.abs(tmp["true"] - tmp["pred_q50"])
        tmp["sq_error_p50"] = (tmp["true"] - tmp["pred_q50"]) ** 2

        hourly_report = (
            tmp.groupby("hour", as_index=False)
            .agg(
                n=("true", "size"),
                mae_p50=("abs_error_p50", "mean"),
                rmse_p50=("sq_error_p50", lambda s: float(np.sqrt(np.mean(s)))),
                coverage_10_90=("in_10_90", "mean"),
                width_10_90=("width_10_90", "mean"),
            )
        )

        hourly_path = report_dir / "hourly_metrics.csv"
        hourly_report.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    print("[CNN1D][eval]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved={metrics_path}")
    print(f"saved={report_csv_path}")
    if hourly_path is not None:
        print(f"saved={hourly_path}")
