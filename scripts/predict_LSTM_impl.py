import os
import yaml
import torch
import torch.nn as nn
import lightning.pytorch as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 

from pv_forecast.common.paths import make_run_dir, copy_config, save_json
from pv_forecast.data.build_df_full import build_df_full

class LSTMForecaster(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()

        self.quantiles = [0.1, 0.5, 0.9]
        self.output_dim = output_dim
        self.num_quantiles = len(self.quantiles)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_dim * self.num_quantiles)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Direct multi-horizon head: one encoder window predicts all H future steps.
        y_hat = self.fc(out[:, -1, :])  # (B, H * Q)
        y_hat = y_hat.view(x.size(0), self.output_dim, self.num_quantiles)
        return y_hat  # (B, H, Q)

    def pinball_loss(self, y_hat, y):
        """
        y_hat: (B, H, Q)
        y:     (B, H)
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if y_hat.ndim != 3:
            raise ValueError(f"Expected y_hat shape (B,H,Q), got {tuple(y_hat.shape)}")

        if y.ndim != 2:
            raise ValueError(f"Expected y shape (B,H), got {tuple(y.shape)}")

        if y_hat.size(1) != y.size(1):
            raise ValueError(
                f"Horizon mismatch: y_hat H={y_hat.size(1)}, y H={y.size(1)}"
            )

        qs = y_hat.new_tensor(self.quantiles).view(1, 1, -1)
        y = y.unsqueeze(-1)
        errors = y - y_hat
        loss = torch.maximum(qs * errors, (qs - 1.0) * errors)
        return loss.mean()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.pinball_loss(y_hat, y)

        y_p50 = y_hat[:, :, 1]
        mae = torch.mean(torch.abs(y_p50 - y))
        rmse = torch.sqrt(torch.mean((y_p50 - y) ** 2))

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_rmse", rmse, prog_bar=True)

        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        return {
            "y_hat": y_hat.detach().cpu(),  # (B,H,Q)
            "y": y.detach().cpu(),          # (B,H)
            "pred_q10": y_hat[:, :, 0].detach().cpu(),
            "pred_q50": y_hat[:, :, 1].detach().cpu(),
            "pred_q90": y_hat[:, :, 2].detach().cpu(),
        }

def main(config_path: str, ckpt_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + "_pred_quantile")
    copy_config(config_path, run_dir)

    df_full = build_df_full(cfg["data"])
    if cfg["data"].get("index_is_time", False):
        df_full = df_full.reset_index()
    
    target = cfg["features"]["target"]
    features = cfg["features"]["input_reals"]
    window_size = cfg["lstm"]["window_size"]
    output_dim = cfg["lstm"]["horizon"]
    time_col = "timestamp" if "timestamp" in df_full.columns else df_full.columns[0]

    n = len(df_full)

    cutoff = int(n * cfg["split"]["train_ratio"])

    train_end = int(n * cfg["split"]["train_ratio"])
    val_end = int(n * (cfg["split"]["train_ratio"] + cfg["split"]["val_ratio"]))

    scaler_x = StandardScaler().fit(df_full.iloc[:cutoff][features])
    scaler_y = StandardScaler().fit(df_full.iloc[:cutoff][[target]])
    
    df_full_scaled = df_full.copy()
    df_full_scaled[features] = scaler_x.transform(df_full[features])
    df_full_scaled[target] = scaler_y.transform(df_full[[target]])

    # val_start_idx = max(0, train_end - window_size)
    test_start_idx = max(0, val_end - window_size)

    # df_val_input = df_full_scaled.iloc[val_start_idx:val_end].copy()
    df_test_input = df_full_scaled.iloc[test_start_idx:].copy()
    
    test_scaled_features = df_test_input[features].values
    test_scaled_target = df_test_input[target].values 
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    model_horizon = int(model.output_dim)
    if model_horizon != output_dim:
        raise ValueError(
            f"Config horizon={output_dim} does not match checkpoint horizon={model_horizon}."
        )

    predictions_scaled, actuals_scaled, rows = [], [], []

    print(f"Predicting quantiles on {device}...")
    with torch.no_grad():
        for i in range(len(test_scaled_features) - window_size - output_dim + 1):
            x_input = torch.FloatTensor(test_scaled_features[i : i + window_size]).unsqueeze(0).to(device)
            y_hat = model(x_input)
            y_hat_np = y_hat.cpu().numpy()[0]  # (H, Q)

            y_true_np = test_scaled_target[i + window_size : i + window_size + output_dim]  # (H,)

            predictions_scaled.append(y_hat_np) 
            actuals_scaled.append(y_true_np)
            forecast_start = df_test_input[time_col].iloc[i + window_size]

            for h in range(output_dim):
                rows.append({
                    "sample": i,
                    "forecast_start": forecast_start,
                    "horizon_idx": h + 1,
                    "timestamp": df_test_input[time_col].iloc[i + window_size + h],
                })

    preds_all = np.array(predictions_scaled) 
    acts_all = np.array(actuals_scaled)

    if preds_all.size == 0:
        raise ValueError("No prediction windows were generated. Check test length, window_size, and horizon.")

    pred_q10_kw = scaler_y.inverse_transform(preds_all[:, :, 0].reshape(-1, 1)).flatten()
    pred_q50_kw = scaler_y.inverse_transform(preds_all[:, :, 1].reshape(-1, 1)).flatten()
    pred_q90_kw = scaler_y.inverse_transform(preds_all[:, :, 2].reshape(-1, 1)).flatten()
    actual_kw = scaler_y.inverse_transform(acts_all.reshape(-1, 1)).flatten()

    res_df = pd.DataFrame(rows)
    res_df["timestamp"] = pd.to_datetime(res_df["timestamp"])
    res_df["forecast_start"] = pd.to_datetime(res_df["forecast_start"])
    res_df["hour"] = res_df["timestamp"].dt.hour
    res_df["true"] = actual_kw
    res_df["pred_q10"] = pred_q10_kw
    res_df["pred_q50"] = pred_q50_kw
    res_df["pred_q90"] = pred_q90_kw
    res_df["pred_q50_scaled"] = preds_all[:, :, 1].reshape(-1)

    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        res_df[col] = res_df[col].clip(lower=0)

    res_df["width"] = res_df["pred_q90"] - res_df["pred_q10"]

    one_step_df = (res_df
        .groupby("timestamp", as_index=False)
        .agg(
            true=("true", "first"),
            pred_q10=("pred_q10", "mean"),
            pred_q50=("pred_q50", "mean"),
            pred_q90=("pred_q90", "mean"),
        )
    )
    one_step_df["hour"] = one_step_df["timestamp"].dt.hour
    one_step_df["width"] = one_step_df["pred_q90"] - one_step_df["pred_q10"]
    one_step_df["in_80"] = (
        (one_step_df["true"] >= one_step_df["pred_q10"])
        & (one_step_df["true"] <= one_step_df["pred_q90"])
    ).astype(int)

    horizon_metrics_df = (res_df
        .groupby("horizon_idx")
        .apply(lambda g: pd.Series({
            "mae": (g["pred_q50"] - g["true"]).abs().mean(),
            "rmse": np.sqrt(((g["pred_q50"] - g["true"]) ** 2).mean()),
            "coverage_80": ((g["true"] >= g["pred_q10"]) & (g["true"] <= g["pred_q90"])).mean(),
            "mean_width": g["width"].mean(),
        }))
        .reset_index()
    )

    # Keep the historical column order near the front, with multi-horizon metadata added.
    res_df = res_df[[
        "sample",
        "forecast_start",
        "horizon_idx",
        "timestamp",
        "hour",
        "true",
        "pred_q10",
        "pred_q50",
        "pred_q90",
        "width",
        "pred_q50_scaled",
    ]]

    metrics = {}
    denom = np.sum(np.abs(actual_kw))
    # sum_actual = np.sum(actual_kw)
    quantiles = [0.1, 0.5, 0.9]
    q_cols = ["pred_q10", "pred_q50", "pred_q90"]

    print("\n" + "="*55)
    print("PREDICTION EVALUATION (Quantile Metrics)")
    print("-" * 55)

    for q, col in zip(quantiles, q_cols):
        y_true = actual_kw
        y_pred = res_df[col].values
        errors = y_true - y_pred
        # errors = actual_kw - res_df[col].values
        q_losses = np.maximum(q * errors, (q - 1) * errors)
        q_loss_sum = q_losses.sum()
        q_loss_mean = q_losses.mean()
        q_risk = 2 * q_loss_sum / denom if denom > 0 else 0

        metrics[f"quantile_loss_mean_{q}"] = float(q_loss_mean)
        metrics[f"q_risk_{q}"] = float(q_risk)
        
        print(f"Quantile {q:0.1f} | Loss: {q_loss_mean:12.2f} | q-Risk: {q_risk:.4f}")

    # 평균 구간 너비 -> tft랑 모델 비교할 때 좋음..? q90 - q10
    mean_width = res_df["width"].mean()
    metrics["mean_width"] = float(mean_width)
    metrics["horizon_metrics"] = {
        str(int(row["horizon_idx"])): {
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "coverage_80": float(row["coverage_80"]),
            "mean_width": float(row["mean_width"]),
        }
        for _, row in horizon_metrics_df.iterrows()
    }
    print("-" * 55)
    print(f"평균 구간 너비 (Mean Width): {mean_width:.2f} kW")
    print("="*55 + "\n")

    res_df["in_80"] = ((res_df["true"] >= res_df["pred_q10"]) & (res_df["true"] <= res_df["pred_q90"])).astype(int)

    out_csv = run_dir / "tables" / "pred_results.csv"
    out_one_step_csv = run_dir / "tables" / "pred_results_by_timestamp.csv"
    out_horizon_metrics_csv = run_dir / "tables" / "horizon_metrics.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    one_step_df.to_csv(out_one_step_csv, index=False, encoding="utf-8-sig")
    horizon_metrics_df.to_csv(out_horizon_metrics_csv, index=False, encoding="utf-8-sig")

    save_json({"ckpt": ckpt_path, "metrics": metrics}, run_dir / "meta" / "predict_meta.json")

    print(f"결과 저장 완료: {out_csv}")
    print(f"타임스탬프별 평균 결과 저장 완료: {out_one_step_csv}")
    print(f"Horizon별 성능 저장 완료: {out_horizon_metrics_csv}")
    print(f"RUN_DIR: {run_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
