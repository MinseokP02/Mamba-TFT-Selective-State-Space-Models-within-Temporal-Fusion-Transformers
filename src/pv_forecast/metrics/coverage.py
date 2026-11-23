import pandas as pd
import numpy as np

def quantile_loss(y_true, y_pred, q: float) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_true - y_pred
    return np.maximum(q * err, (q - 1.0) * err)

def q_risk(y_true, y_pred, q: float, eps: float = 1e-12) -> float:
    """
    TFT paper style normalized quantile loss:
    q-Risk = 2 * sum(QL) / sum(abs(y_true))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    numerator = 2.0 * np.sum(quantile_loss(y_true, y_pred, q))
    denominator = np.sum(np.abs(y_true)) + eps
    return float(numerator / denominator)

def hourly_coverage(one: pd.DataFrame, h_from: int = 6, h_to: int = 18) -> pd.Series:
    hourly = (one.groupby("hour")["in_80"].mean() * 100).reindex(range(24))
    return hourly.loc[h_from:h_to].round(1)

def global_metrics(one: pd.DataFrame) -> dict:
    y = one["true"].to_numpy(dtype=float)
    p10 = one["pred_q10"].to_numpy(dtype=float)
    p50 = one["pred_q50"].to_numpy(dtype=float)
    p90 = one["pred_q90"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(p50 - y)))
    rmse = float(np.sqrt(np.mean((p50 - y) ** 2)))
    coverage = float(np.mean((y >= p10) & (y <= p90)))
    width = float(np.mean(p90 - p10))

    qloss_q10 = float(np.mean(quantile_loss(y, p10, 0.1)))
    qloss_q50 = float(np.mean(quantile_loss(y, p50, 0.5)))
    qloss_q90 = float(np.mean(quantile_loss(y, p90, 0.9)))

    qrisk_q10 = q_risk(y, p10, 0.1)
    qrisk_q50 = q_risk(y, p50, 0.5)
    qrisk_q90 = q_risk(y, p90, 0.9)

    return {
        "n": int(len(one)),

        # existing metrics
        "mae_p50": mae,
        "rmse_p50": rmse,
        "coverage_10_90": coverage,
        "width_10_90": width,

        # added quantile metrics
        "qloss_q10": qloss_q10,
        "qloss_q50": qloss_q50,
        "qloss_q90": qloss_q90,

        # TFT-style normalized metrics
        "qrisk_q10": qrisk_q10,
        "qrisk_q50": qrisk_q50,
        "qrisk_q90": qrisk_q90,
    }