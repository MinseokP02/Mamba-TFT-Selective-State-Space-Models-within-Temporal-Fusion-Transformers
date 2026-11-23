def main(config_path: str, ckpt_path: str):
    import yaml
    from pathlib import Path
    import pandas as pd
    from pytorch_forecasting import TemporalFusionTransformer

    from pv_forecast.common.paths import make_run_dir, copy_config, save_json
    from pv_forecast.data.build_df_full import build_df_full
    from pv_forecast.data.dataset import build_training_dataset, build_validation_dataset
    from pv_forecast.post.extract import postprocess_raw_to_frames

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + "_pred")
    copy_config(config_path, run_dir)
    save_json({"ckpt": ckpt_path}, run_dir / "meta" / "predict_meta.json")

    df_full = build_df_full(cfg["data"])
    cutoff = int(len(df_full) * cfg["split"]["train_ratio"])
    df_train = df_full.iloc[:cutoff].copy()

    training_ds = build_training_dataset(df_train, cfg)
    cutoff_time_idx = int(df_full.iloc[cutoff]["time_idx"])
    val_ds = build_validation_dataset(training_ds, df_full, cutoff_time_idx)
    val_loader = val_ds.to_dataloader(train=False, batch_size=cfg["tft"]["batch_size"], num_workers=0)

    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    raw = model.predict(val_loader, mode="raw", return_x=True)

    final_df, one = postprocess_raw_to_frames(raw, df_full)

    out_one = run_dir / "tables" / "pred_one.csv"
    out_all = run_dir / "tables" / "pred_all_samples.csv"
    one.to_csv(out_one, index=False, encoding="utf-8-sig")
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")

    print("RUN_DIR:", run_dir)
    print("SAVED:", out_one)
    print("SAVED:", out_all)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)