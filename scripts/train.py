
def main(config_path: str):
    import yaml
    import torch
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    from pv_forecast.common.seed import seed_all
    from pv_forecast.common.paths import make_run_dir, copy_config, save_json
    from pv_forecast.data.build_df_full import build_df_full
    from pv_forecast.data.dataset import build_training_dataset, build_validation_dataset
    from pv_forecast.models.tft import build_tft
    import warnings
    warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    )

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"])
    copy_config(config_path, run_dir)

    df_full = build_df_full(cfg["data"])

    cutoff = int(len(df_full) * cfg["split"]["train_ratio"])
    df_train = df_full.iloc[:cutoff].copy()

    training_ds = build_training_dataset(df_train, cfg)
    cutoff_time_idx = int(df_full.iloc[cutoff]["time_idx"])
    val_ds = build_validation_dataset(training_ds, df_full, cutoff_time_idx)

    train_loader = training_ds.to_dataloader(train=True, batch_size=cfg["tft"]["batch_size"], num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=cfg["tft"]["batch_size"], num_workers=0)

    model = build_tft(training_ds, cfg)

    ckpt_dir = f'{cfg["run"]["artifacts_dir"]}/{cfg["run"]["name"]}'
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, save_top_k=1, monitor="val_loss", mode="min", filename="{epoch}-{val_loss:.4f}")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        ckpt_cb
    ]

    trainer = pl.Trainer(
        max_epochs=cfg["tft"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        default_root_dir=str(run_dir),
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    save_json(
        {"best_ckpt": best_ckpt, "run_dir": str(run_dir), "cutoff": cutoff, "cutoff_time_idx": cutoff_time_idx},
        run_dir / "meta" / "train_meta.json"
    )

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)