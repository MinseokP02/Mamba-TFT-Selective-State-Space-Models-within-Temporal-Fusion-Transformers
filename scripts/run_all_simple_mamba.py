from __future__ import annotations

import argparse
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pv_forecast.common.io import ensure_dir
from pv_forecast.common.paths import copy_config, make_run_dir, save_json
from pv_forecast.common.seed import seed_all
from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.metrics.coverage import global_metrics, hourly_coverage
from pv_forecast.models.simple_mamba import (
    SimpleMambaLightningModule,
    parameter_counts,
)
from scripts.run_all_lstm import (
    LSTMKnownFutureDataModule,
    _postprocess_predictions,
    _resolve_split,
    _unique_keep_order,
)
from sklearn.preprocessing import StandardScaler


def _build_datamodule(
    cfg: dict, df_scaled, split: dict, model_feature_cols: list[str],
    known_future_cols: list[str],
) -> LSTMKnownFutureDataModule:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    return LSTMKnownFutureDataModule(
        df_full=df_scaled,
        model_feature_cols=model_feature_cols,
        known_future_cols=known_future_cols,
        target_col=cfg["features"]["target"],
        group_col=data_cfg.get("group_col", "series_id"),
        time_idx_col=data_cfg.get("time_idx_col", "time_idx"),
        encoder_length=cfg["model"]["encoder_length"],
        prediction_length=cfg["model"]["prediction_length"],
        training_cutoff=split["training_cutoff"],
        validation_cutoff=split["validation_cutoff"],
        test_start_time_idx=split["test_start_time_idx"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        allow_missing_time_idx=data_cfg.get("allow_missing_time_idx", False),
    )


def _build_model(cfg: dict, input_size: int) -> SimpleMambaLightningModule:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    return SimpleMambaLightningModule(
        input_size=input_size,
        prediction_length=model_cfg["prediction_length"],
        quantiles=train_cfg.get("quantiles", [0.1, 0.5, 0.9]),
        d_model=model_cfg.get("d_model", 64),
        num_mamba_layers=model_cfg.get("num_mamba_layers", 2),
        d_state=model_cfg.get("d_state", 16),
        d_conv=model_cfg.get("d_conv", 4),
        expand=model_cfg.get("expand", 2),
        dropout=model_cfg.get("dropout", 0.1),
        learning_rate=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        use_future_known_inputs=True,
    )


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    quantiles = train_cfg.get("quantiles", [0.1, 0.5, 0.9])
    if quantiles != [0.1, 0.5, 0.9]:
        raise ValueError("Simple Mamba evaluation currently expects [0.1, 0.5, 0.9].")

    run_dir = make_run_dir(
        cfg["run"]["reports_dir"], cfg["run"]["name"] + "_simple_mamba_all"
    )
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"
    for path in (run_dir, ckpt_dir, table_dir, meta_dir):
        ensure_dir(path)
    copy_config(config_path, run_dir)

    df_full = build_df_full(data_cfg)
    if data_cfg.get("index_is_time", False):
        df_full = df_full.reset_index()
        if "timestamp" not in df_full.columns:
            df_full = df_full.rename(columns={df_full.columns[0]: "timestamp"})
    split = _resolve_split(cfg, df_full)
    target_col = cfg["features"]["target"]
    known_reals = list(cfg["features"]["known_reals"])
    unknown_reals = list(cfg["features"].get("unknown_reals", [target_col]))
    time_idx_col = data_cfg.get("time_idx_col", "time_idx")
    # Keep the indexing column unscaled for window construction while exposing
    # an independently scaled copy to the model, as TFT does internally.
    if time_idx_col in known_reals:
        time_feature_col = f"{time_idx_col}_feature"
        df_full[time_feature_col] = df_full[time_idx_col].astype(float)
        known_reals = [
            time_feature_col if col == time_idx_col else col for col in known_reals
        ]
    model_feature_cols = _unique_keep_order(known_reals + unknown_reals)
    # TFT decoder receives every known real (calendar/time plus forecast
    # weather). Unknown future channels such as pv_output remain zero.
    known_future_cols = [col for col in known_reals if col != target_col]
    required = model_feature_cols + known_future_cols
    missing = [col for col in required if col not in df_full.columns]
    if missing:
        raise ValueError(f"Missing Simple Mamba input columns: {missing}")

    # Fit all normalization statistics strictly on the training partition.
    scaler = StandardScaler()
    scaler.fit(df_full.iloc[: split["train_cutoff"]][model_feature_cols])
    df_scaled = df_full.copy()
    df_scaled[model_feature_cols] = scaler.transform(df_full[model_feature_cols])
    target_idx = model_feature_cols.index(target_col)

    dm = _build_datamodule(
        cfg, df_scaled, split, model_feature_cols, known_future_cols
    )
    model = _build_model(cfg, input_size=len(model_feature_cols) + 1)

    dm.setup("fit")
    xb, yb, _ = next(iter(dm.train_dataloader()))
    validation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(validation_device)
    xb_device = xb.to(validation_device)
    projected = model.model.input_projection(xb_device)
    with torch.no_grad():
        y_hat = model(xb_device).cpu()
    expected = (xb.size(0), cfg["model"]["prediction_length"], len(quantiles))
    assert tuple(projected.shape) == (xb.size(0), xb.size(1), model.hparams.d_model)
    assert tuple(y_hat.shape) == expected
    total_params, trainable_params = parameter_counts(model)
    print(f"input={tuple(xb.shape)} target={tuple(yb.shape)}")
    print(f"projected={tuple(projected.shape)} prediction={tuple(y_hat.shape)}")
    print(f"parameters total={total_params} trainable={trainable_params}")

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=train_cfg.get("early_stop_patience", 10),
                mode="min",
            ),
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        default_root_dir=str(run_dir),
        deterministic=train_cfg.get("deterministic", "warn"),
        gradient_clip_val=train_cfg.get("gradient_clip_val"),
    )
    trainer.fit(model, datamodule=dm)

    best_ckpt = checkpoint.best_model_path or checkpoint.last_model_path
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / "last.ckpt")
        trainer.save_checkpoint(best_ckpt)

    # Loading the selected checkpoint here is both the inference path and a
    # checkpoint round-trip validation.
    best_model = SimpleMambaLightningModule.load_from_checkpoint(best_ckpt)
    dm.setup("test")
    predictor = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        logger=False,
        enable_checkpointing=False,
    )
    outputs = predictor.predict(best_model, dataloaders=dm.test_dataloader())
    pred_all, pred_one = _postprocess_predictions(
        outputs, df_full, scaler, target_idx, quantiles,
        time_idx_col=time_idx_col,
        timestamp_col="timestamp",
    )
    pred_all.to_csv(table_dir / "pred_results.csv", index=False, encoding="utf-8-sig")
    pred_one.to_csv(
        table_dir / "pred_results_by_timestamp.csv", index=False, encoding="utf-8-sig"
    )

    horizon = (
        pred_all.groupby("horizon_idx", as_index=False)
        .agg(mae=("abs_err", "mean"), mse=("sq_err", "mean"))
    )
    horizon["rmse"] = horizon["mse"] ** 0.5
    horizon.to_csv(table_dir / "horizon_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = global_metrics(pred_one)
    metrics["qloss_macro"] = float(
        (metrics["qloss_q10"] + metrics["qloss_q50"] + metrics["qloss_q90"]) / 3
    )
    metrics["split"] = {
        key: split[key]
        for key in (
            "train_ratio", "val_ratio", "test_ratio", "train_cutoff_time_idx",
            "val_cutoff_time_idx",
        )
    }
    save_json(metrics, meta_dir / "metrics.json")
    save_json(
        {
            "best_ckpt": best_ckpt,
            "model_architecture": "causal_mamba_known_future_per_horizon",
            "input_shape": list(xb.shape),
            "output_shape": list(y_hat.shape),
            "model_feature_cols": model_feature_cols,
            "known_future_cols": known_future_cols,
            "uses_future_known_inputs": True,
            "uses_static_covariates": False,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "n_train_samples": len(dm.train_dataset),
            "n_val_samples": len(dm.val_dataset),
            "n_test_samples": len(dm.test_dataset),
        },
        meta_dir / "train_meta.json",
    )
    hourly = hourly_coverage(pred_one, cfg["eval"]["hours_from"], cfg["eval"]["hours_to"])
    hourly.rename("coverage_pct").reset_index().rename(
        columns={"index": "hour"}
    ).to_csv(table_dir / "hourly_coverage.csv", index=False, encoding="utf-8-sig")

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print("METRICS:", meta_dir / "metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
