def main(config_path: str, run_dir: str):
    import yaml
    from pathlib import Path
    import pandas as pd
    from pv_forecast.metrics.coverage import hourly_coverage, global_metrics
    from pv_forecast.common.paths import save_json
    
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    run_dir = Path(run_dir)

    one_path = run_dir / "tables" / "pred_one.csv"
    one = pd.read_csv(one_path, parse_dates=["timestamp"])

    hourly = hourly_coverage(one, cfg["eval"]["hours_from"], cfg["eval"]["hours_to"])
    hourly_df = hourly.rename("coverage_pct").reset_index().rename(columns={"index": "hour"})
    hourly_df.to_csv(run_dir / "tables" / "hourly_coverage.csv", index=False, encoding="utf-8-sig")

    metrics = global_metrics(one)
    save_json(metrics, run_dir / "meta" / "metrics.json")

    # 콘솔에도 출력
    for h, v in hourly.items():
        if pd.isna(v):
            continue
        print(f"{h:02d}:00 | {v:5.1f}%")

    print("SAVED:", run_dir / "tables" / "hourly_coverage.csv")
    print("SAVED:", run_dir / "meta" / "metrics.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)  # predict에서 나온 run_dir
    args = ap.parse_args()
    main(args.config, args.run_dir)