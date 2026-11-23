import yaml
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pv_forecast.common.paths import save_json

def main(config_path: str, run_dir: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    run_dir = Path(run_dir)

    pred_path = run_dir / "tables" / "pred_results.csv"
        
    df = pd.read_csv(pred_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    
    h_from = cfg["eval"]["hours_from"]
    h_to = cfg["eval"]["hours_to"]
    df_eval = df[(df['hour'] >= h_from) & (df['hour'] <= h_to)].dropna(subset=['timestamp']).copy()

    target_col_kw = 'pred_q50' 
    actual_col_kw = 'true'

    mae_kw = mean_absolute_error(df_eval[actual_col_kw], df_eval[target_col_kw])
    rmse_kw = np.sqrt(mean_squared_error(df_eval[actual_col_kw], df_eval[target_col_kw]))
    

    captured = df_eval[(df_eval['true'] >= df_eval['pred_q10']) & 
                       (df_eval['true'] <= df_eval['pred_q90'])]
    picp = len(captured) / len(df_eval) * 100


    metrics = {
        "mae_kw": float(mae_kw),
        "rmse_kw": float(rmse_kw),
        "Coverage": float(picp),
        "sample_count": len(df_eval)
    }
    save_json(metrics, run_dir / "meta" / "metrics.json")

    
    print("\n" + "="*55)
    print(f"QUANTILE EVALUATION RESULTS ({h_from:02d}~{h_to:02d}시)")
    print(f"대상 데이터 수: {len(df_eval)}행")
    print("-" * 55)
    print(f" [Main Metric (Q50)]  MAE : {mae_kw:.2f} kW")
    print(f" [Main Metric (Q50)]  RMSE: {rmse_kw:.2f} kW")
    print("-" * 55)
    print(f" [Reliability] PICP (Q10~Q90 범위 내 적중률): {picp:.2f}%")
    print("="*55 + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    main(args.config, args.run_dir)